"""
Convert zero state dict format
"""
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
import torch
import os

# CKPT_FOLDER = '/mnt_out/colehawk/easel/1_9B_mix_prod_candidate_resume/09_06_01_17/epoch=0-step=195000-validation_loss=0.5897_training_loss_step=0.4781.ckpt'
# where is the deepspeed .ckpt folder
CKPT_FOLDER = "tmp/stage_1"
SAVE_LOCATION = os.path.join(CKPT_FOLDER, "full_fp32_state_dict")
MODEL_SAVE_LOCATION = os.path.join(CKPT_FOLDER, "model_fp32_state_dict")
# which model config
convert_zero_checkpoint_to_fp32_state_dict(CKPT_FOLDER, SAVE_LOCATION)

fp32_state_dict = torch.load(SAVE_LOCATION)

# strip out extra "model" for loading
stripped_state_dict = {
    key[len("model.") :]: val for key, val in fp32_state_dict["state_dict"].items()
}

# print(stripped_state_dict.keys())
torch.save(stripped_state_dict, MODEL_SAVE_LOCATION)
