import os
import json
from transformers import AutoModel, AutoConfig
from mstar.utils.hf_utils import get_model_file_from_s3
from mstar.models.model_factory import config_dict, model_class_dict


import torch 
def print_model_para(model):
    value_dict = {}
    for name, param in model.state_dict().items():
        value = torch.sum(param)        
        value_dict[name] = value   
    return value_dict


def from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs):
    """Load model from s3 bucket or local file.
       - Assume access to the mstar-models s3 bucket.
       - Assume internal model type is named as "[source]-[modeltype]-[modelsize]", e.g. mstar-gpt2-600M
       
    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                - A string, the *model id* of a predefined tokenizer hosted inside a model repo on huggingface.co.
                  Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                  user or organization name, like `dbmdz/bert-base-german-cased`.
                - A path to a *directory* containing vocabulary files required by the tokenizer, for instance saved
                  using the [`~tokenization_utils_base.PreTrainedTokenizerBase.save_pretrained`]
                  method, e.g., `./my_model_directory/`.
        force_download (bool): whether redownload the mstar model files from s3.

    Returns:
        [type]: model
    """
    # args used for loading mstar models
    # Shared args among mstar and hf models.
    force_download = kwargs.get("force_download", None)
    revision = kwargs.get("revision", 'main')
    model = None
    if os.path.isdir(pretrained_model_name_or_path):
        # Load config file.
        config_path = f"{pretrained_model_name_or_path}/config.json"
        with open(config_path, encoding='utf-8') as infile:
            model_config = json.load(infile)
        model_type = model_config.get("model_type", None)
        if model_type in config_dict:
            AutoConfig.register(model_type, config_dict[model_type])
            AutoModel.register(config_dict[model_type], model_class_dict[model_type])
            print(f"Loading mstar model from {pretrained_model_name_or_path}")
            return AutoModel.from_pretrained(pretrained_model_name_or_path)
        else:
            # Load HF models
            print(f"Loading huggingface model from {pretrained_model_name_or_path}")
            return AutoModel.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
    else:
        model_type = "-".join(pretrained_model_name_or_path.split("-")[:2])
        key = pretrained_model_name_or_path
        if model_type not in config_dict:
            # Load HF models
            print(f"Loading huggingface model {key}")
            return AutoModel.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        # Register custom model class
        AutoConfig.register(model_type, config_dict[model_type])
        AutoModel.register(config_dict[model_type], model_class_dict[model_type])    

        # Downlaod from s3. 
        downloaded_folder = get_model_file_from_s3(key, revision, force_download=force_download)
        print(f"Loading mstar model {key}")
        model = AutoModel.from_pretrained(downloaded_folder)  
    return model