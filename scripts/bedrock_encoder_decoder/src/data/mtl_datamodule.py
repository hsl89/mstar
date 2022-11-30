import os
import array
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import numpy as np
import mmap

import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
import sys

sys.path.append("")
from collators import t5_collator
import transformers as hf
import datasets
import hashlib
import math

def read_sample_int16(stream, sample_size, index, np_dtype):
    byte_data = stream[2 * index * sample_size : 2 * (index + 1) * sample_size]
    return np.array(array.array("H", byte_data), dtype=np_dtype)


class MMapDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, sample_size, max_seq_length, max_ip_seq_len=None, max_op_seq_len=None, collator=None, prepend_clm_token=False):
        self.sample_size = sample_size #This corresponds to the EXPANDED_PILE_SIZE for pile, and for P3 this corresponds to max_ip_len + max_op_len
        self.max_seq_length = max_seq_length
        self.max_ip_seq_len = max_ip_seq_len
        self.max_op_seq_len = max_op_seq_len
        self.path = path

        self.tokenizer = tokenizer
        self.pad = tokenizer.pad_token_id
        self.mm = None
        self.len = len(self)
        self.collator = collator #Currently, only used for PILE data
       
        self.prepend_clm_token = prepend_clm_token
        self.clm_token = "<extra_id_0>" 
        self.clm_token_id = self.tokenizer.convert_tokens_to_ids([self.clm_token])[0] 
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.eos_token])[0]
        #If tokenizer vocab size exceeds int16 max value then treat it as uint16 (to make it compatible with example-packed data)
        self.np_dtype = np.uint16 if self.tokenizer.vocab_size > np.iinfo(np.int16).max else np.int16

    def __getitem__(self, index):
        # Hash index with GPU rank to ensure
        # A different data batch per device
        # Does not handle collisions
        rank = torch.distributed.get_rank()
        #index = hash(str(index) + str(rank)) % self.len #This has is non determininstic
        index = int(hashlib.sha256((str(index)+str(rank)).encode()).hexdigest(), 16) % self.len

        if self.mm is None:
            with open(self.path, "r") as f:
                self.mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        return read_sample_int16(self.mm, self.sample_size, index, self.np_dtype)

    def __len__(self):
        with open(self.path, "r") as f:
            mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            size = mm.size() / self.sample_size / 2
            #assert size == int(size)
            return int(size)


class P3Stream(MMapDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        input_ids, labels = np.split(item, [self.max_ip_seq_len]) #Split the array into two parts -- first part is input sequence and second part is output sequence

        if self.np_dtype == np.uint16 :
            #If dtype is uint16, convert it into int32 -- required to later convert to tensor (uint16 --> tensor not allowed)
            input_ids = input_ids.astype(np.int32, casting='safe')
            labels = labels.astype(np.int32, casting='safe')
            
            #If dtype is uint16, the example-packed data uses padding token.
            #Need to replace the padding token with -100 for labels (this will prevent loss calculation for padding tokens)
            labels = np.where(labels == self.pad, -100, labels) #uint16 example-packing is done with 0 as padding token 
        

        decoder_input_ids = t5_collator.shift_tokens_right(
            labels.reshape(1, -1), self.pad, 0
        ).ravel()

        #(TODO): Is this required? Replacing token -100 with self.pad token
        #If this is required, should we do it offline during example-packing to increase speed
        #Should we also dump attention masks offline example-packing?
        out = {
            "input_ids": input_ids,
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
        }

        #For int16 input_ids, replace -100 with padding token
        if self.np_dtype == np.int16 :
            out["input_ids"] = np.where(
                out["input_ids"] == -100, self.pad, out["input_ids"]
            )

        if self.prepend_clm_token:
            out["input_ids"] = self.insert_clm_token(out["input_ids"])

        #Create attention mask to ignore the padding tokens in the input
        #Output padding token (-100) will be ignore during loss calculation 
        out["attention_mask"] = np.where(
            out["input_ids"] == self.pad, 0, 1 #out["input_ids"]
        )

        for key, val in out.items():
            out[key] = torch.LongTensor(val)
        return out

    def insert_clm_token(self, input_ids: np.ndarray):
        
        #Find the last padding index 
        #Assumption: Padding tokens are on the left
        pad_indices = np.where(input_ids== self.pad)[0]
        
        if len(pad_indices) > 0:
        
            #Replace the last padding token with CLM token
            #CLM token should be right before the start of non-padding tokens
            last_pad_ind = pad_indices[-1]
            input_ids[last_pad_ind] = self.clm_token_id
        
        else:
            
            #In case there is no padding, insert CLM token at the start of inputs and drop the last token
            input_ids= np.concatenate(([self.clm_token_id], input_ids[0:-1]))

            #Insert EOS token as the last token for consistency
            input_ids[-1] = self.eos_token_id

        return input_ids


class LabeledDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, labeled_batch, unlabeled_batch, max_seq_length, labeled_max_ip_seq_len, labeled_max_op_seq_len, labeled_data_path, 
                 unlabeled_data_module:pl.LightningDataModule, py_logger):
        super().__init__()
        self.tokenizer = tokenizer
        self.labeled_batch = labeled_batch
        self.unlabeled_batch = unlabeled_batch
        self.max_seq_length = max_seq_length
        self.labeled_max_ip_seq_len = labeled_max_ip_seq_len
        self.labeled_max_op_seq_len = labeled_max_op_seq_len
        self.labeled_data_path = labeled_data_path
        self.unlabeled_data_module = unlabeled_data_module
        self.prepend_clm_token = False
        self.py_logger = py_logger

    def setup(self, stage=None):
        
        if self.unlabeled_batch > 0:
            self.unlabeled_data_module.setup()

        prefix = "data"
        if "KUBERNETES_SERVICE_HOST" in os.environ:
            prefix = "/mnt/"

        self.labeled_train = P3Stream(prefix + self.labeled_data_path + '/training', self.tokenizer, sample_size=self.labeled_max_ip_seq_len+self.labeled_max_op_seq_len,
                                                                                           max_seq_length=self.max_seq_length,
                                                                                           max_ip_seq_len=self.labeled_max_ip_seq_len,
                                                                                           max_op_seq_len=self.labeled_max_op_seq_len,
                                                                                           prepend_clm_token=self.prepend_clm_token)
        self.labeled_val = P3Stream(prefix + self.labeled_data_path + '/valid', self.tokenizer, sample_size=self.labeled_max_ip_seq_len+self.labeled_max_op_seq_len,
                                                                                           max_seq_length=self.max_seq_length,
                                                                                           max_ip_seq_len=self.labeled_max_ip_seq_len,
                                                                                           max_op_seq_len=self.labeled_max_op_seq_len,
                                                                                           prepend_clm_token=self.prepend_clm_token)


    def train_dataloader(self):
        kwargs = {"shuffle": True, "drop_last": True}
        cores = os.cpu_count()

        process_global_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        self.py_logger.info(f"{process_global_rank} with world size {world_size}: P3 batch-size and PILE batch-size {self.labeled_batch}, {self.unlabeled_batch}")
        
        if self.unlabeled_batch > 0 and self.labeled_batch > 0:
            
            #Initialize the self.trainer of unlabeled data module
            if self.unlabeled_data_module.trainer is None: 
                self.unlabeled_data_module.trainer = self.trainer
            
            # Load R samples of Pile for each sample of P3
            unlabeled_dataloader = self.unlabeled_data_module.train_dataloader()
            labeled_dataloader = DataLoader(self.labeled_train,
                            batch_size=self.labeled_batch,
                            num_workers=cores//2,
                            **kwargs) 
            
            self.py_logger.info(f"{process_global_rank} with world size {world_size}: Train Labeled data loader length {len(labeled_dataloader)}")
            self.py_logger.info(f"{process_global_rank} with world size {world_size}: Train Unlabeled data loader length  {len(unlabeled_dataloader)}")

            return pl.trainer.supporters.CombinedLoader([labeled_dataloader, unlabeled_dataloader], mode='max_size_cycle')

        elif self.unlabeled_batch > 0:

            #Initialize the self.trainer of unlabeled data module
            if self.unlabeled_data_module.trainer is None: 
                self.unlabeled_data_module.trainer = self.trainer
            
            unlabeled_dataloader = self.unlabeled_data_module.train_dataloader()
            self.py_logger.info(f"{process_global_rank} with world size {world_size}: Train Unlabeled data loader length  {len(unlabeled_dataloader)}")
            return unlabeled_dataloader,

        elif self.labeled_batch > 0:

            labeled_dataloader = DataLoader(self.labeled_train,
                            batch_size=self.labeled_batch,
                            num_workers=cores,
                            **kwargs) 
            self.py_logger.info(f"{process_global_rank} with world size {world_size}: Train Labeled data loader length {len(labeled_dataloader)}")
            return labeled_dataloader,


    def val_dataloader(self):
        kwargs = {
            "batch_size": self.unlabeled_batch + self.labeled_batch,
            "num_workers": os.cpu_count() // 2,
            "drop_last": True,
        }

        process_global_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        if self.unlabeled_batch > 0 and self.labeled_batch > 0:

            #Initialize the self.trainer of unlabeled data module
            if self.unlabeled_data_module.trainer is None: 
                self.unlabeled_data_module.trainer = self.trainer
            
            unlabeled_dataloader = self.unlabeled_data_module.val_dataloader()
            labeled_dataloader =  DataLoader(self.labeled_val, **kwargs)

            self.py_logger.info(f"{process_global_rank} with world size {world_size}: Valid Labeled data loader length {len(labeled_dataloader)}")
            self.py_logger.info(f"{process_global_rank} with world size {world_size}: Valid Unlabeled data loader length  {len(unlabeled_dataloader)}")

            return   [labeled_dataloader, unlabeled_dataloader]

        elif self.unlabeled_batch > 0:
            #Initialize the self.trainer of unlabeled data module
            if self.unlabeled_data_module.trainer is None: 
                self.unlabeled_data_module.trainer = self.trainer
            
            return   [self.unlabeled_data_module.val_dataloader()]
        
        else:
            return   [DataLoader(self.labeled_val, **kwargs)]

