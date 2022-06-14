import json 
import torch as th
from torch.utils.data import DataLoader, Dataset, SequentialSampler
import pytorch_lightning as pl
from scripts.rlfh.utils.data_utils import compute_indices


def load_jsonl(gcs_path):
    with open(gcs_path, "rb") as f:
        datas = [json.loads(l) for l in f.readlines()]
    return datas

def load_humanfeedback_data(path):
    batch_ids = list(range(3,21)) 
    batch_ids.append(22)
    hf_data = []
    for bid in batch_ids:
        hf_batch_file = f"{path}/batch{bid}.json"
        hf_data.extend([json.loads(line) for line in open(hf_batch_file,'r')])
    return hf_data

class OpenAIFilteredTLDRdataset(Dataset):
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
        return dict(text=text, summary=summary,  
                    input_ids=text_encoding["input_ids"], 
                    attention_mask=text_encoding["attention_mask"].flatten(),
                    labels=labels.flatten(), 
                    labels_attention_mask=summary_encoding["attention_mask"].flatten())
        


class TLDRHumanFeedbackdataset(OpenAIFilteredTLDRdataset):
    def __init__(self, 
                 data, 
                 tokenizer,
                 max_seq_len,
                 summary_max_seq_len):
        super().__init__(data, 
                        tokenizer,
                        max_seq_len,
                        summary_max_seq_len)
    
    def __getitem__(self, index:int):
        sample = self.data[index]
        # text = sample['info']['post']
        choice = int(sample['choice'])
        summary_preferred = sample['summaries'][choice]['text']
        summary_i = sample['summaries'][1-choice]['text']
        text = "summarize: {}".format(sample['info']['post'])
        text_encoding = self.tokenizer(text, 
                                       max_length=self.max_seq_len,
                                       padding="max_length",
                                       truncation=True,
                                       return_attention_mask=True,
                                       add_special_tokens=True,
                                       return_tensors="pt"
                                    )
        summary_i_encoding = self.tokenizer(summary_i, 
                                       max_length=self.summary_max_seq_len,
                                       padding="max_length",
                                       truncation=True,
                                       return_attention_mask=True,
                                       add_special_tokens=True,
                                       return_tensors="pt"
                                    )
        summary_preferred_encoding = self.tokenizer(summary_preferred, 
                                       max_length=self.summary_max_seq_len,
                                       padding="max_length",
                                       truncation=True,
                                       return_attention_mask=True,
                                       add_special_tokens=True,
                                       return_tensors="pt"
                                    )
        summary_i_ids = summary_i_encoding["input_ids"]
        summary_preferred_ids = summary_preferred_encoding["input_ids"]
        summary_i_ids[summary_i_ids==0]= -100
        summary_preferred_ids[summary_preferred_ids==0]= -100
        return dict(text=text, summary_i=summary_i, summary_preferred=summary_preferred,
                    input_ids=text_encoding["input_ids"], 
                    attention_mask=text_encoding["attention_mask"].flatten(),
                    summary_i_ids=summary_i_ids.flatten(), 
                    summary_i_attention_mask=summary_i_encoding["attention_mask"].flatten(),
                    summary_preferred_ids=summary_preferred_ids.flatten(), 
                    summary_preferred_attention_mask=summary_preferred_encoding["attention_mask"].flatten())


class SFTPDataModule(pl.LightningDataModule):
    
    def __init__(self, 
                 tokenizer,
                 logger,
                 train_args):
        super().__init__()
        self.train = load_jsonl(train_args.train_data_path)
        self.train_hf = load_humanfeedback_data(train_args.train_hf_data_path)
        self.valid = load_jsonl(train_args.valid_data_path)
        self.test  = load_jsonl(train_args.test_data_path)
        self.tokenizer   = tokenizer
        self.batch_size  = train_args.batch_size
        self.max_seq_len = train_args.max_seq_len
        self.summary_max_seq_len=train_args.summary_max_seq_len
        self.num_workers=train_args.num_workers
        self.test_batch_size = train_args.test_batch_size
        self.test_distributed_mode = train_args.test_distributed_mode
        self.dataset_name = OpenAIFilteredTLDRdataset
        self.pylogger = logger
    
        
    def setup(self, stage=None):
        self.train_dataset_sl = OpenAIFilteredTLDRdataset(self.train, 
                                                self.tokenizer, 
                                                self.max_seq_len,
                                                self.summary_max_seq_len)
        
        self.train_dataset_hf = TLDRHumanFeedbackdataset(self.train_hf, 
                                                self.tokenizer, 
                                                self.max_seq_len,
                                                self.summary_max_seq_len)
        
        self.valid_dataset = OpenAIFilteredTLDRdataset(self.valid, 
                                            self.tokenizer, 
                                            self.max_seq_len,
                                            self.summary_max_seq_len)
        
    
    def train_dataloader(self):
        
        sl_loader = DataLoader(self.train_dataset_sl, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers)
        hf_loader = DataLoader(self.train_dataset_hf, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers)
        
        loaders = {"sl": sl_loader, "hf": hf_loader}
        return loaders
    
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
            test_set = [self.test[idx] for idx in test_indices]
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