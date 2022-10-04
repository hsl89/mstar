import os
import json 
from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from mstar.utils.hf_utils import get_tokenizer_file_from_s3
from mstar.models.model_factory import config_dict, tokenizer_class_dict, tokenizer_mapping
import logging

logger = logging.getLogger(__name__)



def from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs):
    """Load tokenizer from s3 bucket or local file.
    
    Args:
        pretrained_model_name_or_path ([type]): name of/path to the tokenizer
    """
    # Shared args among mstar and hf models.
    revision = kwargs.pop("revision", 'main')
    cache_dir = kwargs.pop("cache_dir", None)
    if os.path.isdir(pretrained_model_name_or_path):
        # Load config file.
        config_path = f"{pretrained_model_name_or_path}/tokenizer_config.json"
        with open(config_path, encoding='utf-8') as infile:
            tok_config = json.load(infile)
        tokenizer_class = tok_config.get("tokenizer_class", None)
        model_type = tok_config.get("model_type", None)
        if model_type:
            print(f"Loading mstar tokenizer from {pretrained_model_name_or_path}\n")
            # Find model corresponding to this tokenizer., 
            # Caveat: multiple model types could map to the same tokenizer.
            tokenizer_class = tokenizer_mapping[tokenizer_class]
            if issubclass(tokenizer_class, PreTrainedTokenizerFast):
                AutoTokenizer.register(config_dict[model_type], fast_tokenizer_class=tokenizer_class)
            else:
                AutoTokenizer.register(config_dict[model_type], slow_tokenizer_class=tokenizer_class)
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
        else:
            print(f"Loading huggingface tokenizer from {pretrained_model_name_or_path}\n")    
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *inputs, revision=revision, cache_dir=cache_dir, **kwargs)
    else:
        if not isinstance(pretrained_model_name_or_path, str):
            raise ValueError(
                        f"The input must be model id string or path, the current input is {pretrained_model_name_or_path}"
                    )
        model_type = "-".join(pretrained_model_name_or_path.split("-")[:2])
        if model_type in tokenizer_class_dict:
            print(f"Loading mstar tokenizer {pretrained_model_name_or_path}\n")
            downloaded_folder = get_tokenizer_file_from_s3(pretrained_model_name_or_path, revision=revision, cache_dir=cache_dir)
            config_path = f"{downloaded_folder}/tokenizer_config.json"
            with open(config_path, encoding='utf-8') as infile:
                tok_config = json.load(infile)
            tokenizer_class = tok_config.get("tokenizer_class", None)
            assert tokenizer_class, "Invalid tokenizer_config.json: Must have tokenizer_class and model_type specified"
            tokenizer_class = tokenizer_mapping[tokenizer_class]
            if issubclass(tokenizer_class, PreTrainedTokenizerFast):
                AutoTokenizer.register(config_dict[model_type], fast_tokenizer_class=tokenizer_class)
            else:
                AutoTokenizer.register(config_dict[model_type], slow_tokenizer_class=tokenizer_class)
            tokenizer = AutoTokenizer.from_pretrained(downloaded_folder, *inputs, **kwargs)
        else:
            if model_type.startswith("mstar") or model_type.startswith("atm"):
                model_type = "-".join(pretrained_model_name_or_path.split("-")[:2])
                print(f"Loading mstar tokenizer {pretrained_model_name_or_path}\n")
                downloaded_folder = get_tokenizer_file_from_s3(pretrained_model_name_or_path, revision=revision, cache_dir=cache_dir)
                config_path = f"{downloaded_folder}/tokenizer_config.json"
                with open(config_path, encoding='utf-8') as infile:
                    tok_config = json.load(infile)
                tokenizer_class = tok_config.get("tokenizer_class", None)
                assert tokenizer_class, "Invalid tokenizer_config.json: Must have tokenizer_class and model_type specified"
                tokenizer_class = tokenizer_mapping[tokenizer_class]
                if issubclass(tokenizer_class, PreTrainedTokenizerFast):
                    AutoTokenizer.register(config_dict[model_type], fast_tokenizer_class=tokenizer_class)
                else:
                    AutoTokenizer.register(config_dict[model_type], slow_tokenizer_class=tokenizer_class)
                tokenizer = AutoTokenizer.from_pretrained(downloaded_folder, *inputs, **kwargs)
            else:
                print(f"Loading huggingface tokenizer {pretrained_model_name_or_path}\n")
                tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *inputs, revision=revision, cache_dir=cache_dir, **kwargs)
    return tokenizer
