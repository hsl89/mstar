from transformers import T5ForConditionalGeneration
from transformers import T5Config


class MStarT5Config(T5Config):
    model_type = "mstar-t5"


class MStarT5ForConditionalGeneration(T5ForConditionalGeneration):
    config_class = MStarT5Config
