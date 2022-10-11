import time
import os
import sys
import mstar
import collators
import numpy as np
import torch
import transformers
import models
import datasets
import argparse

BATCH_SIZE = 4
DEVICE = "cuda"
MODEL_NAME = "t5-large"

MLM_PROB = 0.165  # chosen to produce multiple of 128 when multiplied by max_length
MEAN_NOISE_SPAN = 3.0
max_length = 512


parser = argparse.ArgumentParser()
parser.add_argument("--preallocate-baddbmm", action="store_true", default=False)
parser.add_argument("--use-baddbmm-score-computation", action="store_true")
parser.add_argument("--use-fused-attention", action="store_true")
parser.add_argument(
    "--softmax-type", type=str, choices=["torch", "mstar_fused"], default="torch"
)
parser.add_argument(
    "--softmax-precision", type=str, choices=["fp16", "bf16"], default="bf16"
)

args = parser.parse_args()

config = transformers.AutoConfig.from_pretrained(MODEL_NAME)

for key, value in vars(args).items():
    setattr(config, key, value)

# cache doesn't work with gradient checkpointing, resetting messes up tflops computation
setattr(config, "use_cache", False)
# necessary for our attention mechanism model
setattr(config, "use_fused_attention", True)

rng_state = torch.get_rng_state()
torch.set_rng_state(rng_state)
torch.manual_seed(1)
np.random.seed(1)

my_model = models.t5_model.T5ForConditionalGeneration(config=config)
# same seed should initialize same model
torch.set_rng_state(rng_state)
torch.manual_seed(1)
np.random.seed(1)

# santize custom gelu, can't load this through huggingface
config.dense_act_fn = (
    "gelu" if config.dense_act_fn == "jit_gelu" else config.dense_act_fn
)
hf_model = transformers.T5ForConditionalGeneration(config=config)

# test equal initializations without pretrained weights
for x, y in zip(my_model.parameters(), hf_model.parameters()):
    assert torch.allclose(x, y), "Relative norm difference {}".format(
        (x - y).float().norm().item() / x.float().norm().item()
    )


# use trained weights after verifying init
hf_model = transformers.T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
my_model.load_state_dict(hf_model.state_dict(), strict=True)


my_model.gradient_checkpointing_enable()
hf_model.gradient_checkpointing_enable()

# need eval to avoid dropout altering the gradients
my_model.eval()
hf_model.eval()

hf_model.bfloat16().to(DEVICE)
my_model.bfloat16().to(DEVICE)

test_data = datasets.arrow_dataset.Dataset.from_file(
    "/mnt/colehawk/pile_no_youtube/val_packed_chunksize_2600.arrow"
)["text"]

tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

# set up MLM span corruption collator
(
    expanded_inputs_length,
    target_length,
) = collators.t5_collator.compute_input_and_target_lengths(
    inputs_length=max_length,
    noise_density=MLM_PROB,
    mean_noise_span_length=MEAN_NOISE_SPAN,
)

collator = collators.unpacked_t5_collator.UnpackedT5DataCollatorForSpanCorruption(
    tokenizer=tokenizer,
    noise_density=MLM_PROB,
    mean_noise_span_length=MEAN_NOISE_SPAN,
    expandend_inputs_length=expanded_inputs_length,
    input_length=max_length,
    target_length=target_length,
    decoder_start_token_id=my_model.config.decoder_start_token_id,
)


# test outputs using this sample batch
batch = collator(test_data[:BATCH_SIZE]).to(DEVICE)

my_outputs = my_model(**batch)
hf_outputs = hf_model(**batch)
try:
    assert torch.allclose(my_outputs.loss, hf_outputs.loss)
except:
    print(
        "Outputs fail, my loss {:.2f} hf loss {:.2f}".format(
            my_outputs.loss.item(), hf_outputs.loss.item()
        )
    )

# do backprop on each loss
my_outputs.loss.backward()
hf_outputs.loss.backward()

# test grads
diffs = []
for x, y in zip(hf_model.parameters(), my_model.parameters()):
    # needs weaker standards than the default due to fused softmax difference
    # megatron tests by taking means
    try:
        assert torch.allclose(x.grad, y.grad)
    except:
        diff = (x.grad - y.grad).abs().reshape(-1).double()

        # if 0 need to divide by anything other than 0
        relative_vals = torch.where(
            diff > 0.0, diff / x.grad.reshape(-1).abs().double(), 0.0
        )
        print("Grads fail max abs diff ", diff.max())
        print("Grads fail max relative diff ", relative_vals.max())
        diffs.append((relative_vals.max(), diff.max()))

print("Top 10 elementwise absolute/relative differences")
for x in list(sorted(diffs[-10:])):
    print("Relative {:.4f} abs {:.4f}".format(x[0].item(), x[1].item()))

print("Loss comparison")
print(my_outputs.loss, hf_outputs.loss)
