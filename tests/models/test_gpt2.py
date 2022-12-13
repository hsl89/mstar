import numpy as np
import torch
import transformers
import mstar
from mstar.models.gpt2 import MStarGPT2Config, MStarGPT2LMHeadModel
import random

DEVICE = "cuda"
# smallest gpt2 model with 124M parameters
MODEL_NAME = "gpt2"


def test_config():
    """Test that M* config accepts all default HF config parameters"""
    # ok to load different transformers versions
    EXEMPT_KEYS = ["transformers_version", "positional_embedding"]
    hf_config = transformers.AutoConfig.from_pretrained(MODEL_NAME)
    hf_config_dict = hf_config.to_dict()

    # add position emebedding parameter for mstar model, as its not available in hf_config
    # default "positional_embedding" for hf gpt2 is "absolute"
    hf_config_dict.update({"positional_embedding": "absolute"})
    mstar_config = MStarGPT2Config(**hf_config_dict)

    # mstar config has additional values, make sure it inherits everything
    for key in hf_config_dict:
        if hasattr(hf_config, key) and key not in EXEMPT_KEYS:
            mstar_val = getattr(mstar_config, key)
            hf_val = getattr(hf_config, key)
            assert (
                mstar_val == hf_val
            ), f"Failed key {key} hf value {hf_val} mstar value {mstar_val}"


def test_loss():
    """Test that M* model output for gpt2 is equal to HF output
    Test that gradients are close to HF gradients.
    """
    # gradient tolerance
    GRADS_ATOL = 1e-5

    samples = [
        "This is a unit test for decoder model .",
        "Unit tests are a waste of time, don't you agree ?",
    ]

    # use trained weights from pretrained gpt2 model from HF
    hf_model = transformers.GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    hf_model.gradient_checkpointing_enable()

    mstar_config = MStarGPT2Config(**hf_model.config.to_dict())
    # update the config to match the huggingface version
    mstar_config.update(
        {
            "fused_scaled_masked_softmax": False,
            "fused_gelu": False,
            "gradient_checkpointing": True,
            "precision": 32,
            "positional_embedding": "absolute"
        }
    )
    mstar_model = MStarGPT2LMHeadModel(config=mstar_config)
    mstar_model.load_state_dict(hf_model.state_dict(), strict=True)

    # need eval to avoid dropout altering the gradients
    mstar_model.eval()
    hf_model.eval()

    hf_model.to(DEVICE)
    mstar_model.to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # set up input batch
    batch = tokenizer(samples, return_attention_mask=True, return_token_type_ids=False,
                    padding="max_length", truncation=True, max_length=64,
                    return_tensors="pt")
    
    batch["labels"] = batch["input_ids"].clone()
    batch["labels"] = batch["labels"] * batch["attention_mask"] + \
                        (1 - batch["attention_mask"]) * -100
    batch.to(DEVICE)

    mstar_outputs = mstar_model(**batch)
    hf_outputs = hf_model(**batch)

    # test that mstar/hf models have equal outputs
    assert torch.allclose(mstar_outputs.loss, hf_outputs.loss)

    # do backprop on each loss
    mstar_outputs.loss.backward()
    hf_outputs.loss.backward()

    # test grads
    for x, y in zip(hf_model.parameters(), mstar_model.parameters()):
        assert torch.allclose(x.grad, y.grad, atol=GRADS_ATOL)


def test_generate():
    """Test that M* gpt2 model generation is equal to HF gpt2 generation
    """
    hf_model = transformers.GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    hf_model.gradient_checkpointing_enable()

    mstar_config = MStarGPT2Config(**hf_model.config.to_dict())
    mstar_config.update(
        {
            "fused_scaled_masked_softmax": False,
            "fused_gelu": False,
            "gradient_checkpointing": True,
            "precision": 32,
            "positional_embedding": "absolute"
        }
    )
    mstar_model = MStarGPT2LMHeadModel(config=mstar_config)
    mstar_model.load_state_dict(hf_model.state_dict(), strict=True)

    # need eval to avoid dropout altering the gradients
    mstar_model.eval()
    hf_model.eval()

    # T5 model was trained in bf16, use this for testing
    hf_model.to(DEVICE)
    mstar_model.to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    text_in = ["This is a unit test"]

    tokens_in = tokenizer(
        text_in, truncation=False, padding=False, return_tensors="pt"
    )
    tokens_in.to(DEVICE)

    hf_model.eval()
    hf_tokens_out = hf_model.generate(
        tokens_in["input_ids"], max_length=50, no_repeat_ngram_size=2, 
        early_stopping=True, num_beams=1
    )

    mstar_model.eval()
    mstar_tokens_out = mstar_model.generate(
        tokens_in["input_ids"], max_length=50, no_repeat_ngram_size=2, 
        early_stopping=True, num_beams=1
    )

    hf_text_out = tokenizer.batch_decode(hf_tokens_out, skip_special_tokens=True)
    mstar_text_out = tokenizer.batch_decode(mstar_tokens_out, skip_special_tokens=True)

    assert mstar_text_out == hf_text_out

