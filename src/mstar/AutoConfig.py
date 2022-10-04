import os
import json
from transformers import AutoConfig, AutoModel
from mstar.utils.hf_utils import get_config_file_from_s3
from mstar.models.model_factory import config_dict, model_class_dict


def from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs):
    """Instantiate one of the configuration classes of the library from s3 bucket, local file or huggingface.co.

       - Assume access to the mstar-models s3 bucket.
       - Assume internal model type is named as "[source]-[model_type]", e.g. mstar-gpt2
       
    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                - A string, the *model id* of a predefined model configuration hosted inside a model repo on
                  huggingface.co or S3://mstar-models. Valid model ids can be located at the root-level, like
                  `bert-base-uncased`, or namespaced under a user or organization name, like
                  `dbmdz/bert-base-german-cased`, or mstar model id format [source]-[model_type]-[model_size], like
                  `mstar-gpt2-672M`.
                - A path to a *directory* containing configuration file saved using the
                  [`~PretrainedConfig.save_pretrained`] method, or the [`~PreTrainedModel.save_pretrained`] method,
                  e.g., `./my_model_directory/`.
                - A path or url to a saved configuration JSON *file*, e.g.,
                  `./my_model_directory/configuration.json`.
        force_download (bool): whether redownload the mstar model config file from s3.

    Returns:
        [type]: config
    """
    # args used for loading mstar models
    # Shared args among mstar and hf models.
    force_download = kwargs.pop("force_download", None)
    revision = kwargs.pop("revision", 'main')
    cache_dir = kwargs.pop("cache_dir", None)
    config = None
    if os.path.isdir(pretrained_model_name_or_path):
        # Load config file.
        config_path = f"{pretrained_model_name_or_path}/config.json"
        with open(config_path, encoding='utf-8') as infile:
            model_config = json.load(infile)
        model_type = model_config.get("model_type", None)
        if model_type in config_dict:
            AutoConfig.register(model_type, config_dict[model_type])
            AutoModel.register(config_dict[model_type], model_class_dict[model_type])
            print(f"Loading mstar config from {pretrained_model_name_or_path}")
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        else:
            print(f"Loading huggingface config from {pretrained_model_name_or_path}")
            return AutoConfig.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
    else:
        model_type = "-".join(pretrained_model_name_or_path.split("-")[:2])
        key = pretrained_model_name_or_path
        if model_type not in config_dict:
            if model_type.startswith("mstar") or model_type.startswith("atm"):
                downloaded_folder = get_config_file_from_s3(key, revision, force_download=force_download, cache_dir=cache_dir)
                print(f"Loading mstar config {key}")
                config_path = f"{downloaded_folder}/config.json"
                with open(config_path, encoding='utf-8') as infile:
                    model_config = json.load(infile)
                return AutoConfig.from_pretrained(downloaded_folder, *inputs, **kwargs)
            else:
                # Load HF configs
                print(f"Loading huggingface config from {pretrained_model_name_or_path}")
                return AutoConfig.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        # Register custom model class
        AutoConfig.register(model_type, config_dict[model_type])
        AutoModel.register(config_dict[model_type], model_class_dict[model_type])    

        # Downlaod from s3. 
        downloaded_folder = get_config_file_from_s3(key, revision, force_download=force_download, cache_dir=cache_dir)
        print(f"Loading mstar config {key}")
        config = AutoConfig.from_pretrained(downloaded_folder, *inputs, **kwargs)  
    return config
