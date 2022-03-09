from mstar.models.gpt2 import MStarGPT2Config, MStarGPT2LMHeadConfig, MStarGPT2Model, MStarGPT2LMHeadModel
from mstar.tokenizers.gptbart import GPT2BartTokenizer

config_dict = {
    "mstar-gpt2": MStarGPT2Config,
    "mstar-gpt2LMHead": MStarGPT2LMHeadConfig
}

model_class_dict = {
    "mstar-gpt2": MStarGPT2Model,
    "mstar-gpt2LMHead": MStarGPT2LMHeadModel
}

tokenizer_class_dict = {
    "mstar-gpt2": GPT2BartTokenizer,
    "mstar-gpt2LMHead": GPT2BartTokenizer
}

tokenizer_class_to_id_dict = {}
for key, value in tokenizer_class_dict.items():
    tokenizer_class_to_id_dict[value.__name__] = key
        
        