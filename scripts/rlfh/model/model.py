import pytorch_lightning as pl
import torch as th
from transformers.optimization import AdamW, get_scheduler
import nltk
nltk.download('punkt')
from torchmetrics.text.rouge import ROUGEScore
import numpy as np


class PlModel(pl.LightningModule):
    def __init__(
        self,
        model,
        tokenizer,
        logger,
        model_args,
        data_args
    ):
        super().__init__()
        self.model = model 
        vocab_size = len(tokenizer)
        if model.config.vocab_size > vocab_size:
            logger.warning(f"model vocab size {model.config.vocab_size} neq tokenizer vocab size {vocab_size}. resizing\n")
            self.model.resize_token_embeddings(vocab_size) # take first len(tokenizer) token embeddings
        self.tokenizer = tokenizer
        self.optimizer_cfg = model_args.OptimizerArgs
        self.max_seq_len = data_args.max_seq_len
        self.summary_max_seq_len= data_args.summary_max_seq_len
        self.batch_size = data_args.batch_size
        self.rouge = ROUGEScore()
        self.rouge_scores = []
        self.data_args = data_args
        self.pylogger = logger
        
    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        output = self.model(input_ids, attention_mask=attention_mask, labels=labels, decoder_attention_mask=decoder_attention_mask)
        return output.loss, output.logits
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        input_ids = input_ids.view(-1, input_ids.size()[-1]) 
        # reshape fro [batch_size, 1, max_seq_len] to [batch_size, max_seq_len]
        attention_mask = batch["attention_mask"]
        label_attention_mask = batch["labels_attention_mask"]
        labels = batch["labels"] 
        outputs = self.model(input_ids=input_ids, 
                                   attention_mask=attention_mask, 
                                   decoder_attention_mask=label_attention_mask,
                                   labels=labels)
        # loss = outputs.loss
        num_nonpad_tokens = th.sum(labels[..., 1:].contiguous() != -100)
        num_nonpad_tokens = num_nonpad_tokens.double()
        loss = outputs.loss * num_nonpad_tokens

        th.distributed.all_reduce(loss)
        th.distributed.all_reduce(num_nonpad_tokens)
        loss = loss / num_nonpad_tokens
        
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        input_ids = input_ids.view(-1, input_ids.size()[-1]) 
        attention_mask = batch["attention_mask"]
        label_attention_mask = batch["labels_attention_mask"]
        labels = batch["labels"] 
        outputs = self.model(input_ids=input_ids, 
                                   attention_mask=attention_mask, 
                                   decoder_attention_mask=label_attention_mask,
                                   labels=labels)
        loss = outputs.loss
        self.log("validation_loss", loss, prog_bar=True,  batch_size=self.batch_size)
        return loss 
    
    def _generation(self, batch):
        input_ids = batch['input_ids']
        input_ids = input_ids.view(-1, input_ids.size()[-1]) 
        attention_mask = batch['attention_mask']
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.summary_max_seq_len,
            num_beams=self.data_args.num_beams,
            repetition_penalty=self.data_args.repetition_penalty,
            length_penalty=self.data_args.length_penalty,
            early_stopping=self.data_args.early_stopping)
        gen_texts = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        pred_summary = list(map(str.strip, gen_texts))  # TODO
        target_summary = batch['summary']
        return pred_summary, target_summary
    
    def parse_score(self, result):
        return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}
    
    def test_step(self, batch, batch_idx):
        pred_summary, target_summary = self._generation(batch)
        scores = list(map(self.rouge, pred_summary, target_summary))
        self.rouge_scores.extend(scores)
        return scores
    
    def test_epoch_end(self, outputs):    
        rouge_dict = {}
        num_test_sample = th.tensor(len(self.rouge_scores)).to(self.device)
        rouge_type = ['rouge1', 'rouge2', 'rougeL']
        rouge_metrics = ['fmeasure', 'precision', 'recall']
        th.distributed.all_reduce(num_test_sample) # get totoal number of test samples.
        for rt in rouge_type:
            for rm in rouge_metrics:
                rank_rouge = th.tensor(np.sum([item[rt + "_" + rm] for item in self.rouge_scores])).to(self.device)
                th.distributed.all_reduce(rank_rouge) # sum of rouge score of all test samples.
                rouge_dict[rt + "_" + rm] = rank_rouge/num_test_sample * 100
        self.log_dict(rouge_dict, rank_zero_only=True)
        return rouge_dict
    
    def configure_optimizers(self):
        learning_rate = self.optimizer_cfg.learning_rate
        return AdamW(self.parameters(), learning_rate)
    