import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Union, Callable, Optional

## Inference ##
def hf_model_inference_fn(
    model: nn.Module,
    batch_idx: int,
    batch: Dict,
    pooling_strategy: str,
    rank: Union[None, int] = None,
) -> Tensor:

    x, m = batch["input_ids"].to(rank), batch["attention_mask"].to(rank)

    with torch.no_grad():
        y = model(input_ids=x, attention_mask=m)
    last_hidden_states = y["last_hidden_state"]
    if pooling_strategy == "cls":
        v = last_hidden_states[:, 0, :]
    elif pooling_strategy == "weighted_mean":
        # sgpt
        weights = (
            torch.arange(start=1, end=last_hidden_states.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(last_hidden_states.size())
            .float()
            .to(rank)
        )
        input_mask_expanded = m.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(
            last_hidden_states * input_mask_expanded * weights, dim=1
        )
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)
        
        # avoid divid by zero if 1 sentence consists on only padding tokens (empty sentence)
        sum_mask[sum_mask==0] = 1
        v = sum_embeddings / sum_mask
    elif pooling_strategy == "mean":
        m = m.float()
        v = last_hidden_states * torch.unsqueeze(
            m, dim=-1
        )  # (batch_size, seq length, embd dim)
        v = torch.sum(v, dim=1)  # (batch size, embd)
        npd = torch.sum(
            m, dim=1, keepdim=True
        )  # (batch size, embd dim) number of non-padding tokens
        # avoid dividing by 0
        npd = torch.where(npd != 0.0, npd, torch.ones_like(npd, dtype=torch.float32))
        # print("v shape: ", v.shape, "npd shape: ", npd.shape)
        v /= npd
        
    return v


def mstar_inference_fn(
    model: nn.Module,
    batch_idx: int,
    batch: Dict,
    pooling_strategy: str,
    rank: Union[None, int] = None,
) -> Tensor:
    device = "cuda:%s" % rank
    x, m = batch["input_ids"].to(device), batch["attention_mask"].to(device)
    with torch.no_grad():
        valid_length = m.sum(axis=1)
        v = model(x, embedding_mode=True, valid_length=valid_length)
    return v
