import torch as th
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import pytorch_lightning as pl
from torch.utils.data import SequentialSampler
from scripts.rlfh.utils.data_utils import compute_indices

class TLDRdataset(Dataset):
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
        text = sample["content"]
        summary = sample["summary"]
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
        return dict(text=text, summary=summary,  
                    input_ids=text_encoding["input_ids"], 
                    attention_mask=text_encoding["attention_mask"].flatten(),
                    labels=labels.flatten(), 
                    labels_attention_mask=summary_encoding["attention_mask"].flatten())


class CNNDMdataset(Dataset):
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
        text = "summarize: {}".format(sample["article"])
        summary = sample["highlights"]
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
        return dict(text=text, summary=summary,  
                    input_ids=text_encoding["input_ids"], 
                    attention_mask=text_encoding["attention_mask"].flatten(),
                    labels=labels.flatten(), 
                    labels_attention_mask=summary_encoding["attention_mask"].flatten())
dataset_name_dict = {
    "ccdv/cnn_dailymail":CNNDMdataset, 
    "reddit":TLDRdataset
}


class PlDataModule(pl.LightningDataModule):
    
    def __init__(self, 
                 tokenizer,
                 logger,
                 train_args):
        super().__init__()      
        dataset_name = train_args.dataset_name
        test_dataset_name = train_args.test_dataset_name
        if dataset_name == "reddit":
            reddit_dataset = load_dataset(dataset_name, split='train')
            reddit_dataset_split = reddit_dataset.train_test_split(test_size=0.01, seed=0)
            train_dataset = reddit_dataset_split['train']
            valid_dataset_split = reddit_dataset_split['test'].train_test_split(test_size=0.5, seed=0)
            valid_dataset = valid_dataset_split['train'] 
            assert test_dataset_name == "reddit"
            test_dataset = valid_dataset_split['test']
        elif dataset_name == "ccdv/cnn_dailymail":
            train_dataset = load_dataset(dataset_name, '3.0.0', split='train')
            valid_dataset = load_dataset(dataset_name, '3.0.0', split='validation')
        else:
            raise(f"{dataset_name} is not a supported dataset")
        
        if test_dataset_name == "ccdv/cnn_dailymail":
            test_dataset = load_dataset(test_dataset_name, '3.0.0', split='test')
        else:
            raise(f"{dataset_name} is not a supported dataset") 
        
        self.train = train_dataset
        self.valid = valid_dataset
        self.test  = test_dataset
        self.tokenizer   = tokenizer
        self.batch_size  = train_args.batch_size
        self.max_seq_len = train_args.max_seq_len
        self.summary_max_seq_len=train_args.summary_max_seq_len
        self.num_workers=train_args.num_workers
        self.test_batch_size = train_args.test_batch_size
        self.dataset_name = dataset_name_dict.get(train_args.dataset_name, TLDRdataset)
        self.test_distributed_mode = train_args.test_distributed_mode
        self.pylogger = logger
        
    def setup(self, stage=None):
        self.train_dataset = self.dataset_name(self.train, 
                                                self.tokenizer, 
                                                self.max_seq_len,
                                                self.summary_max_seq_len)
        
        self.valid_dataset = self.dataset_name(self.valid, 
                                            self.tokenizer, 
                                            self.max_seq_len,
                                            self.summary_max_seq_len)
        
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, 
                          shuffle=False,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers)
    
    def test_dataloader(self):
        if self.test_distributed_mode:
            process_global_rank = th.distributed.get_rank() if th.distributed.is_initialized() else 0
            world_size = th.distributed.get_world_size()
            test_indices = compute_indices(world_size, process_global_rank, len(self.test))
            test_set = self.test.select(test_indices)
        else:
            test_set = self.test
        
        self.pylogger.info(f"Test set size on rank {process_global_rank}:  {len(test_set)}\n")
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
        

