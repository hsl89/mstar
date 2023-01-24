import torch
import os
import transformers
import pytest
from mstar.models.gpt2 import MStarGPT2Config, MStarGPT2LMHeadModel
from mstar.models.gpt2_model import GPT2Attention
from numpy.testing import assert_allclose

CUR_DIR = os.path.dirname(os.path.realpath(__file__))


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


@pytest.mark.parametrize('N_CTX', [256, 512])
@pytest.mark.parametrize('BATCH', [1, 2])
@pytest.mark.parametrize('H', [1, 4])
@pytest.mark.parametrize('D_HEAD', [64, 128])
@pytest.mark.parametrize('bias_shape', ['bhqk', 'b11k', None])
@pytest.mark.parametrize('use_head_mask', [True, False])
# CI machine g4dn.16xlarge doesn't support bf16
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
@pytest.mark.parametrize('cross_attn', [True, False])
@pytest.mark.parametrize('use_alibi', [True, False])
def test_gpt2_flash_attention(N_CTX, BATCH, H, D_HEAD, bias_shape, use_head_mask, use_alibi, dtype, cross_attn):
    if use_alibi and not cross_attn:
        return

    atol, rtol = 1E-5, 1E-5
    if use_alibi:
        rtol = 1E-3
    
    if dtype == torch.float16 or dtype == torch.bfloat16:
        atol, rtol = 1E-1, 1

    device = 'cuda'
    torch.manual_seed(1234)
    q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True) / 10.0
    k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True) / 10.0
    v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True) / 10.0

    if cross_attn:
        if bias_shape == 'bhqk':
            bias = torch.randint(0, 2, (BATCH, H, N_CTX, N_CTX), dtype=dtype, device=device, requires_grad=True).to(dtype)
        elif bias_shape == 'b11k':
            bias = torch.randint(0, 2, (BATCH, 1, 1, N_CTX), dtype=dtype, device=device, requires_grad=True).to(dtype)
        else:
            bias = None
    else:
        if bias_shape == 'bhqk':
            bias = torch.ones((BATCH, H, N_CTX, N_CTX), dtype=dtype, device=device, requires_grad=True).to(dtype)
        elif bias_shape == 'b11k':
            bias = torch.ones((BATCH, 1, 1, N_CTX), dtype=dtype, device=device, requires_grad=True).to(dtype)
        else:
            bias = None

    head_mask = None
    if use_head_mask:
        head_mask = torch.randint(0, 2, (BATCH, H, )).to(dtype).to(device).unsqueeze(-1).unsqueeze(-1).expand((BATCH, H, -1, -1)).to(dtype)

    gpt2_config = transformers.AutoConfig.from_pretrained(CUR_DIR + "/gpt2.json")
    setattr(gpt2_config,'fused_scaled_masked_softmax', False)
    setattr(gpt2_config,'xformers_flash_attention', False)
    if dtype == torch.float32:
        setattr(gpt2_config,'precision', 32) # softmax precision
    elif dtype == torch.float16:
        setattr(gpt2_config,'precision', 16) # softmax precision
    elif dtype == torch.bfloat16:
        setattr(gpt2_config,'precision', 'bf16') # softmax precision
    setattr(gpt2_config,'n_ctx', N_CTX) # seq len
    setattr(gpt2_config,'n_positions', N_CTX) # seq len
    setattr(gpt2_config,'n_head', H)
    setattr(gpt2_config,'scale_attn_weights', True)
    setattr(gpt2_config,'n_embed', H*D_HEAD)
    setattr(gpt2_config,'hidden_size', H*D_HEAD)
    setattr(gpt2_config,'positional_embedding', 'absolute')
    if use_alibi:
        setattr(gpt2_config,'positional_embedding', 'alibi')

    gpt_attn = GPT2Attention(gpt2_config, cross_attn).to('cuda')

    output_torch, _ = gpt_attn._attn_torch(q, k, v, attention_mask=bias, head_mask=head_mask)
    output_xops = gpt_attn._attn_xformers_flash_attn(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), \
        attention_mask=bias, head_mask=head_mask.permute(0, 2, 1, 3) if head_mask is not None else head_mask)

    assert_allclose(output_torch.to(torch.float32).cpu().detach().numpy(), output_xops.to(torch.float32).cpu().detach().numpy(), atol, rtol)

    grad_out = torch.randn(BATCH, H, N_CTX, D_HEAD, dtype=dtype, device="cuda")
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()
    output_torch.backward(grad_out, retain_graph=True)

    q_grad, k_grad, v_grad = q.grad, k.grad, v.grad


    q.grad = None
    k.grad = None
    v.grad = None

    output_xops.backward(grad_out, retain_graph=True)

    q_xformer_grad, k_xformer_grad, v_xformer_grad = q.grad, k.grad, v.grad

    if bias is not None and bias.requires_grad:
        bias_xformer_grad = bias.grad

    assert_allclose(q_grad.to(torch.float32).cpu().detach().numpy(), q_xformer_grad.to(torch.float32).cpu().detach().numpy(), atol, rtol)
    assert_allclose(k_grad.to(torch.float32).cpu().detach().numpy(), k_xformer_grad.to(torch.float32).cpu().detach().numpy(), atol, rtol)
    assert_allclose(v_grad.to(torch.float32).cpu().detach().numpy(), v_xformer_grad.to(torch.float32).cpu().detach().numpy(), atol, rtol)
