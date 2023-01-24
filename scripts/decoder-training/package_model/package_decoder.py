import json
import os
import sys
from pathlib import Path
import argparse
import torch
from mstar.models.gpt2 import MStarGPT2Config, MStarGPT2LMHeadConfig, MStarGPT2Model, MStarGPT2LMHeadModel
import transformers
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)

# single shard for M* models due to download assumptions
MAX_SHARD_SIZE = "999GB"


def load_model_checkpoints(path, model):
    state_dict = torch.load(path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if "module" in state_dict:
        state_dict = state_dict["module"]
    unwrapped_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module"):  # deepspeed zero_to_fp32.py
            new_key = ".".join(key.split(".")[2:])
        else:
            new_key = ".".join(key.split(".")[1:])
        unwrapped_state_dict[new_key] = value
    if 'lm_head.weight' not in unwrapped_state_dict.keys():
        unwrapped_state_dict['lm_head.weight'] = unwrapped_state_dict['transformer.wte.weight']
    model.load_state_dict(unwrapped_state_dict)
    model.eval()
    return model

def load_model_checkpoints_nohead(path, model):
    state_dict = torch.load(path, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if "module" in state_dict:
        state_dict = state_dict["module"]
    unwrapped_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module"):  # deepspeed zero_to_fp32.py
            new_key = ".".join(key.split(".")[3:])
        else:
            new_key = ".".join(key.split(".")[2:])
        unwrapped_state_dict[new_key] = value
    model.load_state_dict(unwrapped_state_dict, strict=False)
    model.eval()
    return model

def remove_config_element(model_config):
    # remove some unncessary config
    model_config.pop('task_specific_params', None)
    model_config.pop('id2label', None)
    model_config.pop('label2id', None)
    return model_config


def run(args):

    config_path= args.model_config_path
    with open(config_path) as infile:
        model_config = json.load(infile)

    if args.precision == 'bf16':
        precision = 'bf16'
    elif args.precision == 'fp16':
        precision = 16
    elif args.precision == 'fp32':
        precision = 32

    model_config.update(
        {
            "vocab_size": args.vocab_size,
            "fused_scaled_masked_softmax": True,
            "fused_gelu": True,
            "precision": precision,
            "positional_embedding": args.pos_emb,
        }
    )
    model_config = remove_config_element(model_config)

    if args.sharded_model:
        full_state_dict_save_path = args.ckpt_path+'state_dict.pt'
        convert_zero_checkpoint_to_fp32_state_dict(args.ckpt_path, full_state_dict_save_path)
        # update the ckpt_path to load the model
        args.ckpt_path = full_state_dict_save_path

    # with LM head
    if args.model_head:
        config = MStarGPT2LMHeadConfig(**model_config)
        config.bos_token_id=args.eos_token_id
        config.eos_token_id=args.eos_token_id
        model = MStarGPT2LMHeadModel(config)
        if args.precision == 'bf16':
            model = model.to(torch.bfloat16)
        elif args.precision == 'fp16':
            model = model.to(torch.float16)
        model = load_model_checkpoints(args.ckpt_path, model)
        model.save_pretrained(args.model_save_path, max_shard_size=MAX_SHARD_SIZE)
    else:
        config = MStarGPT2Config(**model_config)
        config.bos_token_id=args.eos_token_id
        config.eos_token_id=args.eos_token_id
        model = MStarGPT2Model(config)
        if args.precision == 'bf16':
            model = model.to(torch.bfloat16)
        elif args.precision == 'fp16':
            model = model.to(torch.float16)
        model = load_model_checkpoints_nohead(args.ckpt_path, model)
        model.save_pretrained(args.model_save_path, max_shard_size=MAX_SHARD_SIZE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Package decoder language models in huggingface style')
    parser.add_argument('--ckpt-path', type=str, default=None,
                        help='path to the trained model checkpoint')
    parser.add_argument('--model-save-path', type=str, default=None,
                        help='HF style packaged model will be saved to this path')
    parser.add_argument('--model-config-path', type=str, default="None",
                        help='model configs path')
    parser.add_argument('--pos-emb', type=str, default='absolute',
                        help='type of position embedding used to train the model')
    parser.add_argument('--vocab-size', type=int, default=None,
                        help='vocabulary size used to train the model')
    parser.add_argument('--input-length', type=int, default=None,
                        help='input sequence length used to train the model')
    parser.add_argument('--eos-token-id', type=int, default=None,
                        help='token_id for eos token, can be different for different tokenizers')
    parser.add_argument('--precision', type=str, default="bf16",
                        help='precision of the pretrained model, should be one [bf16, fp16, fp32]')
    parser.add_argument('--sharded-model', action='store_true',
                        help='whether the model is sharded')
    parser.add_argument('--model-head', action='store_true',
                        help='whether to save language model head')

    args = parser.parse_args()
    if args.precision not in ['bf16', 'fp16', 'fp32']:
        raise ValueError('precision must be one of [bf16, fp16, fp32]')
    run(args)


