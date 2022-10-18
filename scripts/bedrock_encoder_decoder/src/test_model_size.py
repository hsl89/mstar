import models
import torch
import transformers

import models
from torch import nn
from pytorch_lightning.utilities.meta import init_meta_context
import yaml

CONFIG_PATH = "config/model/base.yaml"

with open(CONFIG_PATH, "r") as fp:
    yaml_conf = yaml.load(fp)

print(yaml_conf)

hf_model_config = models.configuration_t5.MStarT5Config(**yaml_conf)
# setattr(hf_model_config,'softmax_precision',16)
# setattr(hf_model_config,'use_fused_softmax',True)
# hf_model_config = transformers.AutoConfig.from_pretrained('t5-11b')
# hf_model_config = transformers.AutoConfig.from_pretrained('config/model/gelu_20B.json')
# hf_model_config = transformers.AutoConfig.from_pretrained('config/model/11B.json')
# hf_model_config = transformers.AutoConfig.from_pretrained('google/ul2')
hf_model_config.save_pretrained("tmp")

print(hf_model_config)
# models.t5_model.T5ForConditional
model = transformers.T5ForConditionalGeneration(config=hf_model_config)
print(
    "Total B parameters {:.2f}".format(
        sum([x.numel() for x in model.parameters()]) / 1000000000
    )
)
"""
with init_meta_context():
    model = transformers.T5ForConditionalGeneration(config=hf_model_config)
    print("Total B parameters {:.2f}".format(sum([x.numel() for x in model.parameters()])/1000000000))
    #print(model)
"""
