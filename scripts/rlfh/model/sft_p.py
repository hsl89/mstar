from scripts.rlfh.model.model import PlModel
from scripts.rlfh.utils.model_utils import get_logp
import torch as th


class SFTPModel(PlModel):
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
        self.loss_beta = model_args.OptimizerArgs.loss_beta
        self.loss_weight = model_args.OptimizerArgs.loss_weight
        self.margin = model_args.OptimizerArgs.hinge_margin
        self.type_hf = model_args.type_hf
    
    def training_step(self, batch, batch_idx):
        # Supervised loss.
        loss = self._compute_sl_loss(batch)
        
        if self.loss_beta > 0:
            # Human feedback loss
            hinge_loss, log_prob_summary_i, log_prob_summary_p = self._compute_hf_loss(batch)
            train_loss = self.loss_weight * loss + self.loss_beta * hinge_loss
            
            log_dict = {
                "train_loss_sl": loss, 
                "hinge_loss_hf": hinge_loss,   
                "training_loss": train_loss,
                "log_prob_summary_i": th.mean(log_prob_summary_i),
                "log_prob_summary_p": th.mean(log_prob_summary_p)
            }
            self.log_dict(log_dict, prog_bar=True, logger=True)
            loss = train_loss 
        else:
            self.log("training_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def _compute_sl_loss(self, batch):
        input_ids = batch["sl"]["input_ids"]
        input_ids = input_ids.view(-1, input_ids.size()[-1]) 
        # Reshape from [batch_size, 1, max_seq_len] to [batch_size, max_seq_len]
        attention_mask = batch["sl"]["attention_mask"]
        label_attention_mask = batch["sl"]["labels_attention_mask"]
        labels = batch["sl"]["labels"] 
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
        return loss
    
    def _compute_hf_loss(self, batch):
        input_ids_hf = batch["hf"]["input_ids"]
        input_ids_hf = input_ids_hf.view(-1, input_ids_hf.size()[-1]) 
        attention_mask_hf = batch["hf"]["attention_mask"]
        summary_i_attention_mask = batch["hf"]["summary_i_attention_mask"]
        summary_i_ids = batch["hf"]["summary_i_ids"] 
        
        summary_preferred_attention_mask = batch["hf"]["summary_preferred_attention_mask"]
        summary_preferred_ids = batch["hf"]["summary_preferred_ids"] 
        
        output_i = self.model(input_ids=input_ids_hf, 
                                attention_mask=attention_mask_hf, 
                                decoder_attention_mask=summary_i_attention_mask,
                                labels=summary_i_ids)
        
        output_p = self.model(input_ids=input_ids_hf, 
                                attention_mask=attention_mask_hf, 
                                decoder_attention_mask=summary_preferred_attention_mask,
                                labels=summary_preferred_ids)
        
        log_prob_summary_i = get_logp(output_i.logits, summary_i_ids, summary_i_attention_mask)
        log_prob_summary_p = get_logp(output_p.logits, summary_preferred_ids, summary_preferred_attention_mask)
        # Processing human feedback tensor here:
        if self.type_hf == "random": 
            random_sign = th.sign(th.rand(log_prob_summary_i.size())*2 - 1).to(self.device)
            log_probs = self.margin + (log_prob_summary_i - log_prob_summary_p) * random_sign
        elif self.type_hf == "negative": 
            log_probs = self.margin - (log_prob_summary_i - log_prob_summary_p)
        else:
            log_probs = self.margin + log_prob_summary_i - log_prob_summary_p 
        log_probs[log_probs<0] = 0
        hinge_loss = th.mean(log_probs)
        th.distributed.all_reduce(hinge_loss)
        return hinge_loss, log_prob_summary_i, log_prob_summary_p