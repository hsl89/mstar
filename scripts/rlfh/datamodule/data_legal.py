import pytorch_lightning as pl
import json
from regex import P 
import torch as th
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import numpy as np
from scripts.rlfh.utils.data_utils import compute_indices
from transformers import AutoModel, AutoTokenizer

def json_data_loader(data_path, post='original_text', summary='reference_summary', uid='uid'):
    with open(data_path, "rb") as f:
        legal_data = json.load(f)
    data = []
    for k, v in legal_data.items():
        sample = {}
        sample['post'] = v.get(post, "")
        sample['summary'] = v.get(summary, "")
        sample['uid'] = v[uid]
        data.append(sample)
    return data

def save_json(data_dict, data_path):
    with open(data_path, 'w') as fp:
        json.dump(data_dict, fp)


class LegalEDTSummarizationDataset(Dataset):
    def __init__(self, 
                 data, 
                 tokenizer,
                 max_seq_len,
                 summary_max_seq_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.summary_max_seq_len = summary_max_seq_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index:int):
        sample = self.data[index]
        summary = sample["summary"]
        text = "summarize: {}".format(sample["post"])
        text_encoding = self.tokenizer(text, 
                                       max_length=self.max_seq_len,
                                       padding="max_length",
                                       truncation=True,
                                       return_attention_mask=True,
                                       add_special_tokens=True,
                                       return_tensors="pt"
                                    )
        summary_encoding = self.tokenizer(summary, 
                                       max_length=self.summary_max_seq_len,
                                       padding="max_length",
                                       truncation=True,
                                       return_attention_mask=True,
                                       add_special_tokens=True,
                                       return_tensors="pt"
                                    )
        labels = summary_encoding["input_ids"]
        labels[labels==0]= -100
        return dict(text=text, summary=summary,  key=str(sample['uid']), orignal_text=sample["post"],
                    input_ids=text_encoding["input_ids"], 
                    attention_mask=text_encoding["attention_mask"].flatten(),
                    labels=labels.flatten(), 
                    labels_attention_mask=summary_encoding["attention_mask"].flatten())
        

class InferenceDataModule(pl.LightningDataModule):
    
    def __init__(self, 
                 tokenizer,
                 logger,
                 data_args):
        super().__init__()
        self.test  = json_data_loader(data_args.test_data_path, 
                                      data_args.post_key, 
                                      data_args.summary_key,
                                      data_args.uid_key)
        self.tokenizer   = tokenizer
        self.batch_size  = data_args.batch_size
        self.max_seq_len = data_args.max_seq_len
        self.summary_max_seq_len=data_args.summary_max_seq_len
        self.num_workers=data_args.num_workers
        self.test_batch_size = data_args.test_batch_size
        self.test_distributed_mode = data_args.test_distributed_mode
        self.pylogger = logger
        self.dataset_name = LegalEDTSummarizationDataset
    
        
    def setup(self, stage=None):
        pass
    
    def test_dataloader(self):
        if self.test_distributed_mode:    
            process_global_rank = th.distributed.get_rank() if th.distributed.is_initialized() else 0
            world_size = th.distributed.get_world_size()
            test_indices = compute_indices(world_size, process_global_rank, len(self.test))
            test_set = [self.test[idx] for idx in test_indices]
            self.pylogger.info(f"Test set size on rank {process_global_rank}:  {len(test_set)}\n")
        else:
            test_set = self.test
            self.pylogger.info(f"Test set size:  {len(test_set)}\n")
        
        
        self.test_dataset = self.dataset_name(test_set, 
                                              self.tokenizer, 
                                              self.max_seq_len,
                                              self.summary_max_seq_len)
        return DataLoader(self.test_dataset,
                          batch_size=self.test_batch_size, 
                          num_workers=self.num_workers,
                          sampler=SequentialSampler(self.test_dataset),
                          shuffle=False,
                          drop_last=False)