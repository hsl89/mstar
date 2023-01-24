## Loading Model ##
import os
from mstar import AutoModel, AutoTokenizer
from mstar.models.m5_models import M5BertForPreTrainingPreLN, M5BertConfig
from mstar.models.t5 import MStarT5Config, T5ForConditionalGeneration
import json

dir_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_path, "config.json"), "r") as f:
    config = json.load(f)

CACHE_DIR = config["model_cache_dir"]

# Note: all models loaded are in eval mode
def load_hf_model(device, model_name, tokenizer_name, debug, use_bfloat16, cache_dir=CACHE_DIR):
    m = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    t = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    if use_bfloat16:
        m = m.bfloat16()
    return m.to(device).eval(), t


def load_mstar_model(device, model_name, tokenizer_name, debug, use_bfloat16, cache_dir=CACHE_DIR):
    if debug:
        config = M5BertConfig(num_hidden_layers=2, use_fused_softmax=True)
        model = M5BertForPreTrainingPreLN(config).to("cuda:%s" % device)
    else:
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    if use_bfloat16:
        model = model.bfloat16()
    t = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=cache_dir)
    return model.to(device).eval(), t


def load_t5(device, model_id, tokenizer_id, revision, debug, use_bfloat16, cache_dir=CACHE_DIR):
    # pull out the encoder
    if debug:
        config = MStarT5Config(
            num_layers=2,
            num_decoder_layers=2,
            use_fused_attention=True,
            softmax_type="mstar_fused",
        )
        m = T5ForConditionalGeneration(config)
    else:
        # cache_dir = os.path.join("/root/.cache/mstar/transformers", model_id, revision)
        m = AutoModel.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
    del m.decoder

    t = AutoTokenizer.from_pretrained(model_id, revision=revision, cache_dir=cache_dir)
    if use_bfloat16:
        m = m.bfloat16()
    return m.encoder.to(device).eval(), t
