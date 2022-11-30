from . import input_perturbers, prompt_perturbers
from .perturber import IdentityPerturber, InputPerturber, PromptPerturber


def get_perturber(key):
    return PERTURBER_MAP[key]

def get_all_perturber_keys():
    return list(PERTURBER_MAP.keys())

def get_all_input_perturber_keys():
    return list(INPUT_PERTURBER_MAP.keys())

def get_all_training_input_perturber_keys():
    return [k for k in INPUT_PERTURBER_MAP.keys() if k != 'ReplaceWithRandomCharacterPerturber']

def get_all_zeroshot_prompt_perturber_keys():
    return [k for k in get_all_prompt_perturber_keys() if k[:7] != 'Example']

def get_all_prompt_perturber_keys():
    return list(PROMPT_PERTURBER_MAP.keys())

def is_abstract(cls):
    return bool(getattr(cls, "__abstractmethods__", False))

def construct_perturber_dict(module, perturber_type):
    perturbers = {}
    for item in module.__dict__.values():
        if isinstance(item, type) and issubclass(item, perturber_type) and not is_abstract(item):
            perturbers[item.__name__] = item
    return perturbers

INPUT_PERTURBER_MAP = construct_perturber_dict(input_perturbers, InputPerturber)
PROMPT_PERTURBER_MAP = construct_perturber_dict(prompt_perturbers, PromptPerturber)

PERTURBER_MAP = {"Original": IdentityPerturber,
                 **INPUT_PERTURBER_MAP,
                 **PROMPT_PERTURBER_MAP,
                }
