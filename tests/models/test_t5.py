import numpy as np
import mstar
import mstar.models.t5
import transformers
import torch

PADDING_STYLE = "max_length"
BATCH_SIZE = 4
DEVICE = "cuda"
MODEL_NAME = "t5-small"
MAX_LENGTH = 32


def test_config():
    """Test that M* config accepts all default HF config parameters"""
    # ok to load different transformers versions
    EXEMPT_KEYS = ["transformers_version"]
    hf_config = transformers.AutoConfig.from_pretrained("t5-small")
    hf_config_dict = hf_config.to_dict()

    mstar_config = mstar.models.t5.MStarT5Config(**hf_config_dict)

    # mstar config has additional values, make sure it inherits everything
    for key in hf_config_dict:
        if hasattr(hf_config, key) and key not in EXEMPT_KEYS:
            mstar_val = getattr(mstar_config, key)
            hf_val = getattr(hf_config, key)
            assert (
                mstar_val == hf_val
            ), f"Failed key {key} hf value {hf_val} mstar value {mstar_val}"

    # mstar config should inherit the default positional embeddings
    assert mstar_config.positional_embedding == "t5"


def test_initialization():
    """Test that HF and M* initialize exactly the same model"""

    config = transformers.AutoConfig.from_pretrained(MODEL_NAME)
    config.num_layers = 2
    config.num_decoder_layers = 2
    config.tie_word_embeddings = True
    # same seed should initialize same model
    rng_state = torch.get_rng_state()
    torch.set_rng_state(rng_state)
    torch.manual_seed(1)
    np.random.seed(1)
    mstar_config = mstar.models.t5.MStarT5Config(**config.to_dict())
    mstar_model = mstar.models.t5.T5ForConditionalGeneration(config=mstar_config)
    torch.set_rng_state(rng_state)
    torch.manual_seed(1)
    np.random.seed(1)
    hf_model = transformers.T5ForConditionalGeneration(config=config)
    # print(hf_model.config.tie_word_embedddings)
    # test equal initializations without pretrained weights
    for x, y in zip(mstar_model.parameters(), hf_model.parameters()):
        assert x.shape == y.shape, "Different shapes {} {}}".format(x.shape, y.shape)
        assert torch.allclose(x, y), "Relative norm difference {} shape {}".format(
            (x - y).float().norm().item() / x.float().norm().item(), x.shape
        )


def test_loss():
    """Test that M* model output for t5-small is equal to HF output
    Test that gradients are close to HF gradients.
    Numerical differences expected due to fused softmax kernel
    """

    # higher abosolute tolerance of gradients
    # due to different softmax implementation
    GRADS_ATOL = 5e-2

    # these mirror the T5 loss function, so we expect non-random loss using pretrained weights
    # Random loss is >=10.0 based on 32k vocab size
    input_sentences = [
        "Hi my name is Cole what's your <extra_id_1>?",
        "Unit tests are <extra_id_1> of time, don't you <extra_id_2> ?",
    ]
    output_targets = ["<extra_id_1> name", "<extra_id_1> a waste of <extra_id_2> agree"]

    # use trained weights from pretrained T5 model
    # pretrained models are bffloat16 so we use this too
    hf_model = transformers.T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    mstar_config = mstar.models.t5.MStarT5Config(**hf_model.config.to_dict())
    mstar_config.softmax_input_precision = "bf16"
    mstar_config.use_fused_softmax = False
    mstar_model = mstar.models.t5.T5ForConditionalGeneration(config=mstar_config)
    mstar_model.load_state_dict(hf_model.state_dict(), strict=True)

    mstar_model.gradient_checkpointing_enable()
    hf_model.gradient_checkpointing_enable()

    # need eval to avoid dropout altering the gradients
    mstar_model.eval()
    hf_model.eval()

    # T5 model was trained in bf16, use this for testing
    hf_model.bfloat16().to(DEVICE)
    mstar_model.bfloat16().to(DEVICE)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

    # set up input batch
    inputs = tokenizer.batch_encode_plus(
        input_sentences, max_length=MAX_LENGTH, padding=PADDING_STYLE
    )
    targets = tokenizer.batch_encode_plus(
        output_targets, max_length=MAX_LENGTH, padding=PADDING_STYLE
    )
    for key, val in inputs.items():
        inputs[key] = torch.stack([torch.tensor(x) for x in val]).long().cuda()

    for key, val in targets.items():
        targets[key] = torch.stack([torch.tensor(x) for x in val]).long().cuda()

    batch = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": torch.where(
            targets["input_ids"] == tokenizer.pad_token_id, -100, targets["input_ids"]
        ),
    }

    mstar_outputs = mstar_model(**batch)
    hf_outputs = hf_model(**batch)

    # test that mstar/hf models have equal outputs
    assert torch.allclose(mstar_outputs.loss, hf_outputs.loss)

    # do backprop on each loss
    mstar_outputs.loss.backward()
    hf_outputs.loss.backward()

    # test grads
    for x, y in zip(hf_model.parameters(), mstar_model.parameters()):
        # needs weaker standards than the default due to fused softmax difference
        assert torch.allclose(x.grad, y.grad, atol=GRADS_ATOL)
