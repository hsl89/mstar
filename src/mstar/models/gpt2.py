from .gpt2_model import GPT2LMHeadModel, GPT2Model, GPT2Config

class MStarGPT2Config(GPT2Config):
    model_type = "mstar-gpt2"
    
class MStarGPT2LMHeadConfig(GPT2Config):
    model_type = "mstar-gpt2LMHead"
    
class MStarGPT2Model(GPT2Model):
    config_class = MStarGPT2Config

class MStarGPT2LMHeadModel(GPT2LMHeadModel):
    config_class = MStarGPT2LMHeadConfig
    
