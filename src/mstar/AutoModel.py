import os
import json
import transformers
from collections import defaultdict
from transformers import AutoModel, AutoConfig
from transformers.models.auto.modeling_auto import (
    MODEL_MAPPING_NAMES,
    MODEL_FOR_PRETRAINING_MAPPING_NAMES,
    MODEL_WITH_LM_HEAD_MAPPING_NAMES,
)
from mstar.utils.hf_utils import get_model_file_from_s3
from mstar.models.model_factory import config_dict, model_class_dict


import torch


def print_model_para(model):
    value_dict = {}
    for name, param in model.state_dict().items():
        value = torch.sum(param)
        value_dict[name] = value
    return value_dict


def create_mappings():
    mapping_dict = defaultdict(dict)
    for model_mapping, auto_func in [
        (MODEL_MAPPING_NAMES, "AutoModel"),
        (MODEL_FOR_PRETRAINING_MAPPING_NAMES, "AutoModelForPreTraining"),
        (MODEL_WITH_LM_HEAD_MAPPING_NAMES, "AutoModelForMaskedLM"),
    ]:
        for model_type, model_class in model_mapping.items():
            mapping_dict[model_type][model_class] = auto_func
    return mapping_dict


def get_auto_function(model_config):
    """Get Huggingface Auto-function from model config
    """
    model_type = model_config.get("model_type", None)
    if not model_type:
        raise ValueError("Please specify model model type in model config")
    architectures = model_config.get("architectures", None)
    if not architectures:
        raise ValueError("Please specify model architectures in model config")
    model_mapping_dict = create_mappings()
    # TODO(zhenghuj): considering multiple architectures for one config
    return getattr(transformers, model_mapping_dict[model_type][architectures[0]])


def from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs):
    """Load model from s3 bucket or local file.
       - Assume access to the mstar-models s3 bucket.
       - Assume internal model type is named as "[source]-[modeltype]-[modelsize]", e.g. mstar-gpt2-600M
       
    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                - A string, the *model id* of a predefined model configuration hosted inside a model repo on
                  huggingface.co or S3://mstar-models. Valid model ids can be located at the root-level, like
                  `bert-base-uncased`, or namespaced under a user or organization name, like
                  `dbmdz/bert-base-german-cased`, or mstar model id format [source]-[model_type]-[model_size], like
                  `mstar-gpt2-672M`.
                - A path to a directory containing model weights saved using
                  [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                - A path or url to a *PyTorch state_dict save file* (e.g, `./pt_model/pytorch_model.bin`). In this
                  case, `from_pt` should be set to `True` and a configuration object should be provided as `config`
                  argument. This loading path is slower than converting the PyTorch model in a TensorFlow model
                  using the provided conversion scripts and loading the TensorFlow model afterwards.
        force_download (bool): whether redownload the mstar model files from s3.
        cache_dir (Union[str, os.PathLike], optional): 
                Path to a directory in which a downloaded pretrained model configuration should be cached if
                the standard cache should not be used.
        device_map (str or Dict[str, Union[int, str, torch.device]], optional):
                A map that specifies where each submodule should go. It doesn’t need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent
                to the same device. To have Accelerate compute the most optimized device_map automatically,
                set device_map="auto".

    Returns:
        [type]: model
    """
    # args used for loading mstar models
    # Shared args among mstar and hf models.
    force_download = kwargs.pop("force_download", None)
    revision = kwargs.pop("revision", "main")
    cache_dir = kwargs.pop("cache_dir", None)
    device_map = kwargs.pop("device_map", None)
    model = None
    if os.path.isdir(pretrained_model_name_or_path):
        # Load config file.
        config_path = f"{pretrained_model_name_or_path}/config.json"
        with open(config_path, encoding="utf-8") as infile:
            model_config = json.load(infile)
        model_type = model_config.get("model_type", None)
        if model_type in config_dict:
            AutoConfig.register(model_type, config_dict[model_type])
            AutoModel.register(config_dict[model_type], model_class_dict[model_type])
            print(f"Loading mstar model from {pretrained_model_name_or_path}")
            model = AutoModel.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs, device_map=device_map)
        else:
            print(f"Loading huggingface model from {pretrained_model_name_or_path}")
            return AutoModel.from_pretrained(
                pretrained_model_name_or_path, *inputs, **kwargs
            )
    else:
        model_type = "-".join(pretrained_model_name_or_path.split("-")[:2])
        key = pretrained_model_name_or_path
        if model_type not in config_dict:
            if model_type.startswith("mstar") or model_type.startswith("atm"):
                downloaded_folder = get_model_file_from_s3(
                    key, revision, force_download=force_download, cache_dir=cache_dir
                )
                print(f"Loading mstar model {key}")
                config_path = f"{downloaded_folder}/config.json"
                with open(config_path, encoding="utf-8") as infile:
                    model_config = json.load(infile)
                return get_auto_function(model_config).from_pretrained(
                    downloaded_folder,
                    *inputs, **kwargs,
                    device_map=device_map
                )
            else:
                # Load HF models
                print(f"Loading huggingface model from {pretrained_model_name_or_path}")
                return AutoModel.from_pretrained(
                    pretrained_model_name_or_path, *inputs, **kwargs
                )
        # Register custom model class
        AutoConfig.register(model_type, config_dict[model_type])
        AutoModel.register(config_dict[model_type], model_class_dict[model_type])

        # Downlaod from s3.
        downloaded_folder = get_model_file_from_s3(
            key, revision, force_download=force_download, cache_dir=cache_dir
        )
        print(f"Loading mstar model {key}")
        model = AutoModel.from_pretrained(downloaded_folder, *inputs, **kwargs, device_map=device_map)  
    return model
