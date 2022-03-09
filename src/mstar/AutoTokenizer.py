import os
import json 
from transformers import AutoTokenizer
from mstar.utils.hf_utils import get_tokenizer_file_from_s3
from mstar.models.model_factory import config_dict, tokenizer_class_dict, tokenizer_class_to_id_dict
import logging

logger = logging.getLogger(__name__)



def from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs):
    """Load tokenizer from s3 bucket or local file.
    
    Args:
        pretrained_model_name_or_path ([type]): name of/path to the tokenizer
    """
    # Shared args among mstar and hf models.
    revision = kwargs.get("revision", 'main')
    if os.path.isdir(pretrained_model_name_or_path):
        # Load config file.
        config_path = f"{pretrained_model_name_or_path}/tokenizer_config.json"
        with open(config_path, encoding='utf-8') as infile:
            tok_config = json.load(infile)
        tokenizer_class = tok_config.get("tokenizer_class", None)
        model_type = tokenizer_class_to_id_dict.get(tokenizer_class, None)
        if model_type:
            print(f"Loading mstar tokenizer from {pretrained_model_name_or_path}\n")
            # Find model corresponding to this tokenizer., 
            # Caveat: multiple model types could map to the same tokenizer.
            AutoTokenizer.register(config_dict[model_type], slow_tokenizer_class=tokenizer_class_dict[model_type])
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        else:
            print(f"Loading huggingface tokenizer from {pretrained_model_name_or_path}\n")    
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
    else:
        if not isinstance(pretrained_model_name_or_path, str):
            raise ValueError(
                        f"The input must be model id string or path, the current input is {pretrained_model_name_or_path}"
                    )
        model_type = "-".join(pretrained_model_name_or_path.split("-")[:2])
        if model_type in tokenizer_class_dict:
            print(f"Loading mstar tokenizer {pretrained_model_name_or_path}\n")            
            AutoTokenizer.register(config_dict[model_type], slow_tokenizer_class=tokenizer_class_dict[model_type])
            downloaded_folder = get_tokenizer_file_from_s3(pretrained_model_name_or_path, revision=revision)
            tokenizer = AutoTokenizer.from_pretrained(downloaded_folder)
        else:
            print(f"Loading huggingface tokenizer {pretrained_model_name_or_path}\n")
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)
    return tokenizer
