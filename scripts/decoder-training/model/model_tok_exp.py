import torch
from transformers import AdamW
from transformers.trainer_pt_utils import get_parameter_names
import pytorch_lightning as pl

from mstar.models.gpt2_model import GPT2Config, GPT2LMHeadModel
from .model_module import PlModel as BasePlModel

from apex.optimizers import FusedLAMB, FusedNovoGrad
from mstar.utils import flops_calc
from mstar.optimizers import FusedAdam
from .optimization import get_scheduler
import time
import deepspeed

# to be moved to package script and import here
def read_state_dict(state_dict_path, model):
    state_dict = torch.load(state_dict_path, map_location="cpu")
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
    return unwrapped_state_dict


class PlModel(BasePlModel):

    def setup(self, stage=None):
        self.model = GPT2LMHeadModel(self.config)
        self.py_logger.info(f"Loaded GPT model config: {self.config}\n")

    def configure_sharded_model(self):
        # if pre-trained model weights are provided
        if self.model_args.init_model_path:
            state_dict = read_state_dict(self.model_args.init_model_path, self.model)
            self.py_logger.info(f"read model weights from: {self.model_args.init_model_path}\n")

            if self.model_args.init_embd_random:
                # get the randomly initialized weights of embedding matrix
                init_embd_weight = self.model.transformer.wte.weight
                state_dict['transformer.wte.weight'] = init_embd_weight
                state_dict['lm_head.weight'] = init_embd_weight
                self.py_logger.info(f"randomly initialized embedding matrix\n")
            # initialize the model with pre-trained weights
            self.model.load_state_dict(state_dict)
            #self.model._load_from_state_dict(state_dict, prefix='')
            self.py_logger.info(f"initialized model weights from: {self.model_args.init_model_path}\n")

        layer_list = []
        # if we only want to train the embedding matrix. 
        if self.model_args.train_embd_only:
            layer_list = ['transformer.wte.weight', 'ln_f', 'lm_head.weight']
        # if we want to train specific layers provided as input
        if self.model_args.train_layers:
            for layer in self.model_args.train_layers:
                layer_list.append('transformer.h.'+str(layer)+'.')
        print('list of trainable layers: {}'.format(layer_list))
        if self.model_args.train_embd_only or self.model_args.train_layers:
            for name, param in self.model.named_parameters():
                if all(layer not in name for layer in layer_list):
                    param.requires_grad = False
                    self.py_logger.info(f"Setting grad false for: {name}\n")
            self.py_logger.info(f"limited training layers to {self.model_args.train_layers}\n")

