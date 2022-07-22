from scripts.rlfh.datamodule.data_tldr import SFTPDataModule, OpenAIFilteredTLDRdataset
from scripts.rlfh.utils.data_utils import compute_indices, load_humanfeedback_data, load_jsonl
import pytorch_lightning as pl

class TLDRAlignedHumanFeedbackdataset(OpenAIFilteredTLDRdataset):
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
        text = "summarize: {}".format(sample['post'])
        groundtruth_summary = sample["summary"]
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
        summary_groundtruth_encoding = self.tokenizer(groundtruth_summary, 
                                       max_length=self.summary_max_seq_len,
                                       padding="max_length",
                                       truncation=True,
                                       return_attention_mask=True,
                                       add_special_tokens=True,
                                       return_tensors="pt"
                                    )
        summary_i_ids = summary_i_encoding["input_ids"]
        summary_preferred_ids = summary_preferred_encoding["input_ids"]
        summary_groundtruth_ids = summary_groundtruth_encoding["input_ids"]
        summary_i_ids[summary_i_ids==0]= -100
        summary_preferred_ids[summary_preferred_ids==0]= -100
        return dict(text=text, summary_i=summary_i, summary_preferred=summary_preferred,
                    summary_groundtruth=groundtruth_summary,
                    input_ids=text_encoding["input_ids"], 
                    attention_mask=text_encoding["attention_mask"].flatten(),
                    summary_i_ids=summary_i_ids.flatten(), 
                    summary_i_attention_mask=summary_i_encoding["attention_mask"].flatten(),
                    summary_preferred_ids=summary_preferred_ids.flatten(), 
                    summary_preferred_attention_mask=summary_preferred_encoding["attention_mask"].flatten(),
                    summary_groundtruth_ids=summary_groundtruth_ids.flatten(),
                    summary_groundtruth_attention_mask=summary_groundtruth_encoding["attention_mask"]
                    )
        
        
class SFTPAlignedDataModule(SFTPDataModule, pl.LightningDataModule):
    
    def __init__(self, 
                 tokenizer,
                 logger,
                 train_args):
        super(SFTPDataModule, self).__init__()
        self.train = load_jsonl(train_args.train_data_path)
        self.train_hf = load_jsonl(train_args.train_hf_data_path)[0]
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
        
        self.train_dataset_hf = TLDRAlignedHumanFeedbackdataset(self.train_hf, 
                                                self.tokenizer, 
                                                self.max_seq_len,
                                                self.summary_max_seq_len)
        
        self.valid_dataset = OpenAIFilteredTLDRdataset(self.valid, 
                                            self.tokenizer, 
                                            self.max_seq_len,
                                            self.summary_max_seq_len)