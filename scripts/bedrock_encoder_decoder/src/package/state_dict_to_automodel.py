"""
Convert zero state dict format
"""
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
import torch
import os
import mstar
import yaml
import torch as th
import mstar.models.t5

CFG_FILE = "../config/model/11B.yaml"
STATE_DICT_PATH = (
    "/mnt_out/colehawk/easel/11B_stage_2_t5/10_13_13_56/last.ckpt/model_fp32_state_dict"
)

hf_model_config = mstar.models.t5.MStarT5Config(**yaml.load(open(CFG_FILE)))

state_dict = th.load(STATE_DICT_PATH, map_location="cpu")

# fine to clear old variable naming, this is a constant buffer
try:
    state_dict.pop("encoder.block.0.layer.0.SelfAttention.alibi_positional_bias")
    state_dict.pop("decoder.block.0.layer.0.SelfAttention.alibi_positional_bias")
except:
    pass

# Need to make sure vocab sizes match, then load from checkpoint
hf_model_config.vocab_size = state_dict["shared.weight"].shape[0]


hf_model_config.gradient_checkpointing = True
hf_model_config.use_cache = False

hf_model_config.softmax_precision = "bf16"
assert False, "Check softmax"
# load model on CPU for more memory
model = mstar.models.t5.MStarT5ForConditionalGeneration(config=hf_model_config).cpu()
assert (
    model.shared.weight.shape[1] == hf_model_config.d_model
), "Safety check since load below is non-strict"
# TODO: also check layer number
# strict=False since PTL does not save embed_tokens.weight for encoder/decoder since they are replicated by the lm_head.weight and shared.weight
model.load_state_dict(state_dict, strict=False)

# one shard since M* autodownload will fail otherwise
model.save_pretrained("pretrained_automodel", max_shard_size="999GB")
