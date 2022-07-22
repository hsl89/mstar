from scripts.rlfh.model.model import PlModel
from scripts.rlfh.utils.data_utils import save_json
import torch as th
import numpy as np


class InferenceModel(PlModel):
    def __init__(
        self,
        model,
        tokenizer,
        logger,
        model_args,
        data_args
    ):
        super().__init__(model,
                tokenizer,
                logger,
                model_args,
                data_args)
        self.policy_name = model_args.policy_name
        self.generated_summary_path = data_args.generated_summary_path
        self.generated_summary = {}
        self.test_distributed_mode = data_args.test_distributed_mode

        
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
    
    def test_step(self, batch, batch_idx):
        pred_summary, target_summary = self._generation(batch)
        scores = list(map(self.rouge, pred_summary, target_summary))
        for idx, item in enumerate(pred_summary):
            self.generated_summary[batch['key'][idx]] = {"original_text": batch['orignal_text'][idx],
                                                "summaries":  [{"text": pred_summary[idx], "policy": self.policy_name},
                                                               {"text": target_summary[idx], "policy": 'reference_summary'},
                                                               ],
                                                "uid": batch['key'][idx]
                                                }
        self.rouge_scores.extend(scores)
        return scores
    
    def test_epoch_end(self, outputs):    
        rouge_dict = {}
        num_test_sample = th.tensor(len(self.rouge_scores)).to(self.device)
        rouge_type = ['rouge1', 'rouge2', 'rougeL']
        rouge_metrics = ['fmeasure', 'precision', 'recall']
        if self.test_distributed_mode:
            th.distributed.all_reduce(num_test_sample) # get totoal number of test samples.
        for rt in rouge_type:
            for rm in rouge_metrics:
                rank_rouge = th.tensor(np.sum([item[rt + "_" + rm] for item in self.rouge_scores])).to(self.device)
                if self.test_distributed_mode:
                    th.distributed.all_reduce(rank_rouge) # sum of rouge score of all test samples.
                rouge_dict[rt + "_" + rm] = rank_rouge/num_test_sample * 100
        self.log_dict(rouge_dict, rank_zero_only=True)
        
        # save dict
        save_json(self.generated_summary, self.generated_summary_path)
        return rouge_dict