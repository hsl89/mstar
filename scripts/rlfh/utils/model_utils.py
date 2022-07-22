import torch as th


def get_logp(logits, summary_i_ids, summary_i_attention_mask):
    m = th.nn.LogSoftmax(dim=2)
    log_prob_i = m(logits) # batch size x seq_len x vocab_size
    idx = th.clone(summary_i_ids.view(-1).unsqueeze(-1))
    idx[idx==-100] = 0 # Padding index is -100, the corresponding logprob won't be selected.
    log_prob_view = log_prob_i.view(-1, log_prob_i.size()[-1])  # [batch size x seq_len, vocab_size]
    log_prob_summary_i = th.gather(log_prob_view, -1, idx).view(log_prob_i.size()[:2])
    log_prob_summary_i *= summary_i_attention_mask  #  batch_size x seq_len
    log_prob_summary_i = th.mean(log_prob_summary_i, dim=1)  # batch_size x 1; average log_prob per token.
    return log_prob_summary_i