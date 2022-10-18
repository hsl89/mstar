from torch.utils.data import Dataset, DataLoader
import os
import math
import copy
import torch
import transformers
from transformers import T5Tokenizer
from dataclasses import dataclass, field
from transformers import (
    T5ForConditionalGeneration,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    AutoConfig,
    T5Config,
)
from transformers import TrainingArguments, HfArgumentParser

# from transformers.models.longformer import LongformerSelfAttention
import argparse
import pyarrow as pa
import warnings

# from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers import T5Tokenizer
import datasets
import torch.nn as nn

warnings.simplefilter("ignore")
NUM_REPEATS = 10
BATCH_SIZE = 4
MAX_STEPS = 5
MLM_PROB = 0.1665
MEAN_NOISE_SPAN = 3.0
EVAL_ONLY = False
MAX_LENGTH = 2048  # 2048#MAX_SEQ_LENGTH #2048
CLM_MAX_DOC = 2
CLM_MIX_RATIO = 0.25
CUTOFF = 256
TRAIN_DATA_PATH = (
    "/mnt/pretraining-data/package-09-22-22-v1/train1_packed_chunksize_2600.arrow"
)
CLM_RATIO = 0.5
INDICES = [91021, 10003, 9299, 791912]  # ,918121,97112,112189,11017]
INDICES = [918121, 97112, 112189, 11017]
assert len(INDICES) == BATCH_SIZE
import numpy as np

torch.manual_seed(1)
np.random.seed(1)


class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_indices):
        self.dataset_indices = dataset_indices

    def __getitem__(self, index):
        pa_index = pa.array(index if type(index) == list else [index])
        return self.dataset_indices.take(pa_index)["text"].to_pylist()

    def __len__(self):
        return len(self.dataset_indices)


train_dataset = datasets.arrow_dataset.Dataset.from_file(TRAIN_DATA_PATH)

batch = train_dataset.select(INDICES)["text"]


total_examples = len(train_dataset)
assert len(INDICES) == BATCH_SIZE
# Check table has single chunk
# https://issues.apache.org/jira/browse/ARROW-11989

model = transformers.T5ForConditionalGeneration.from_pretrained("t5-small")
# 2113 to match bedrock
tokenizer = transformers.AutoTokenizer.from_pretrained("t5-small", extra_ids=2113)

model.resize_token_embeddings(len(tokenizer))
config = transformers.AutoConfig.from_pretrained("t5-base")

import t5_collator


"""
import alexa_tm

data_collator = alexa_tm.Seq2SeqMixDenoisingCollator(
            tokenizer=tokenizer,
            mask_fraction=MLM_PROB,
            max_length=MAX_LENGTH,#expanded_inputs_length,
            mix_ratio=CLM_MIX_RATIO,
            clm_max_doc=CLM_MAX_DOC,
            device='cuda')
data_collator = t5_collator.Seq2SeqMixDenoisingCollator(
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,#expanded_inputs_length,
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=config.decoder_start_token_id,
            noise_density=0.155,#MLM_PROB
            mean_noise_span_length=MEAN_NOISE_SPAN,
            mix_ratio=CLM_MIX_RATIO,
            clm_max_doc=CLM_MAX_DOC,            
            keep_sentinel_ids=False)


data_collator = t5_collator.CLMCollator(
            tokenizer=tokenizer,
            max_input_length=MAX_INPUT_LENGTH,
            max_output_length=MAX_OUTPUT_LENGTH,
            decoder_start_token_id=config.decoder_start_token_id,
            clm_token="<extra_id_0>",
            clm_min_split_ratio=0.2,
            clm_max_split_ratio=0.8,
)
"""
MAX_INPUT_LENGTH = 256  # 2048
MAX_OUTPUT_LENGTH = 128  # 1024

expanded_inputs_length, target_length = t5_collator.compute_input_and_target_lengths(
    inputs_length=MAX_INPUT_LENGTH,
    noise_density=MLM_PROB,
    mean_noise_span_length=MEAN_NOISE_SPAN,
)

print("Inputs lengths", expanded_inputs_length, target_length)
data_collator = t5_collator.MixedT5DataCollatorForSpanCorruption(
    tokenizer=tokenizer,
    noise_density=MLM_PROB,
    mean_noise_span_length=MEAN_NOISE_SPAN,
    expandend_inputs_length=expanded_inputs_length,
    input_length=MAX_INPUT_LENGTH,
    target_length=target_length,  # for T5
    pad_token_id=tokenizer.pad_token_id,
    decoder_start_token_id=config.decoder_start_token_id,
    clm_ratio=CLM_RATIO,
    clm_max_output_length=MAX_OUTPUT_LENGTH,
)

batch = [x for x in batch]

# 0.2498, leave out sentinel ids
# 0.2498, keep in sentinel ids

batch = [x for x in batch]
print("Setup")
import time

t = time.time()
for _ in range(NUM_REPEATS):
    out = data_collator(batch)

print("No prints")
total_time = time.time() - t
per_batch_time = total_time / NUM_REPEATS
print(f"Per-batch collation time {per_batch_time}")

for x in out:
    print(x, out[x].shape)

    assert (
        out[x].shape[0] == BATCH_SIZE
    ), f"For key {x} Sizes {out[x].shape} with batch size {BATCH_SIZE}"

for idx in range(BATCH_SIZE):
    print(20 * "*")
    print(f"Element {idx}")
    print(20 * "*")
    print("Encoder first input_ids \n", out["input_ids"][idx, :CUTOFF])
    print(
        "Encoder first text \n", tokenizer.decode(list(out["input_ids"][idx, :CUTOFF]))
    )
    print("Encoder last input_ids \n", out["input_ids"][idx, -CUTOFF:])
    try:
        # expect  Padding and attention mask to be the same ratios
        non_padding_ratio = (
            sum(out["attention_mask"][idx].reshape(-1))
            / out["attention_mask"][idx].numel()
        )
        print("Encoder non-padding ratio ", non_padding_ratio)
        non_padding_ratio = (
            sum(out["attention_mask"][idx].reshape(-1))
            / out["attention_mask"][idx].numel()
        )
        attn_mask_nonzero_ratio = (
            sum((out["input_ids"][idx] != tokenizer.pad_token_id).reshape(-1))
            / out["attention_mask"][idx].numel()
        )
        print("Encoder attention mask nonzero ratio ", attn_mask_nonzero_ratio)
        assert torch.allclose(non_padding_ratio, attn_mask_nonzero_ratio)
    except:
        warnings.warn("No attention mask?????")

    this_is_clm = "<extra_id_1>" in tokenizer.decode(out["input_ids"][idx, :5])
    if this_is_clm:
        print("Full text")
        print(
            "Encoder last text \n",
            tokenizer.decode(list(out["input_ids"][idx, -CUTOFF:])),
        )
        to_decode = out["labels"][idx, :CUTOFF]
        print(
            "Decoder first target text \n",
            tokenizer.decode(
                list(torch.where(to_decode == -100, tokenizer.pad_token_id, to_decode))
            ),
        )
        print("Decoder first labels \n", out["labels"][idx, :CUTOFF])
        print("Decoder last labels \n", out["labels"][idx, -CUTOFF:])
        # deal with padding tokens conversion to -100
        to_decode = out["labels"][idx, -CUTOFF:]
        print(
            "Decoder last target text \n",
            tokenizer.decode(
                list(torch.where(to_decode == -100, tokenizer.pad_token_id, to_decode))
            ),
        )
        print(
            "Decoder targets ratio",
            sum(out["labels"][idx] != -100) / out["labels"][idx].numel(),
        )
        try:
            print(
                "Decoder input text \n",
                tokenizer.decode(list(out["decoder_input_ids"][idx, :CUTOFF])),
            )
        except:
            pass

    else:
        print(
            "Encoder last text \n",
            tokenizer.decode(list(out["input_ids"][idx, -CUTOFF:])),
        )
        print("Decoder first labels \n", out["labels"][idx, :CUTOFF])
        to_decode = out["labels"][idx, :CUTOFF]
        print(
            "Decoder first target text \n",
            tokenizer.decode(
                list(torch.where(to_decode == -100, tokenizer.pad_token_id, to_decode))
            ),
        )
        # list(out['labels'][idx,:CUTOFF])))
        print("Decoder last labels \n", out["labels"][idx, -CUTOFF:])
        # deal with padding tokens conversion to -100
        to_decode = out["labels"][idx, -CUTOFF:]
        print(
            "Decoder last target text \n",
            tokenizer.decode(
                list(torch.where(to_decode == -100, tokenizer.pad_token_id, to_decode))
            ),
        )
        print(
            "Decoder targets ratio",
            sum(out["labels"][idx] != -100) / out["labels"][idx].numel(),
        )
        try:
            print(
                "Decoder input text \n",
                tokenizer.decode(list(out["decoder_input_ids"][idx, :CUTOFF])),
            )
        except:
            pass
"""
model = model.cuda()
out = out.to('cuda')
with torch.no_grad():
    loss = model(**out).loss

print("loss ",loss)

#loss.backward()

"""

for i, x in enumerate(batch):
    print(f"Example {i}")
    print(x)
