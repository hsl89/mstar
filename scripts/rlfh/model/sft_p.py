from scripts.rlfh.model.model import PlModel
import torch as th

def get_logp(logits, summary_i_ids, summary_i_attention_mask):
    m = th.nn.LogSoftmax(dim=2)
    log_prob_i = m(logits) # batch size x seq_len x vocab_size
    idx = th.clone(summary_i_ids.view(-1).unsqueeze(-1))
    idx[idx==-100] = 0 # Padding index -100, the corresponding logprob won't be selected.
    log_prob_view = log_prob_i.view(-1, log_prob_i.size()[-1])  # [batch size x seq_len, vocab_size]
    log_prob_summary_i = th.gather(log_prob_view, -1, idx).view(log_prob_i.size()[:2])
    log_prob_summary_i *= summary_i_attention_mask  #  batch_size x seq_len
    log_prob_summary_i = th.mean(log_prob_summary_i, dim=1)  # batch_size x 1; average log_prob per token.
    return log_prob_summary_i


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
    
    def training_step(self, batch, batch_idx):
        # Supervised loss.
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
        
        if self.loss_beta > 0:
            # Human feedback loss
            input_ids_hf = batch["hf"]["input_ids"]
            input_ids_hf = input_ids_hf.view(-1, input_ids_hf.size()[-1]) 
            attention_mask_hf = batch["hf"]["attention_mask"]
            summary_i_attention_mask = batch["hf"]["summary_i_attention_mask"]
            summary_i_ids = batch["hf"]["summary_i_ids"] 
            
            summary_preferred_attention_mask = batch["hf"]["summary_i_attention_mask"]
            summary_preferred_ids = batch["hf"]["summary_i_ids"] 
            
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
            log_probs = self.margin + log_prob_summary_i - log_prob_summary_p 
            log_probs[log_probs<0] = 0
            hinge_loss = th.mean(log_probs)
            th.distributed.all_reduce(hinge_loss)
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