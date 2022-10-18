import os
import array
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import numpy as np
import mmap

import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
import sys

sys.path.append("")
from collators import t5_collator
import transformers as hf
import datasets

# import consts

PILE_EXPANDED_SIZE = 600
# Size of all sequence samples when they go into the model
SAMPLE_SIZE = 512
# ID of the pad token
PAD_TOKEN = -100
METADATA = {"ACL": "bucket-owner-full-control"}
"""
# T5 Tokenizer. Can edit and rerun processing with other
# tokenizers as long as they have the same API
_ = TOKENIZER()
EOS = _.encode(_.eos_token)
SEP = EOS + EOS
SEP_STR = _.eos_token + _.eos_token
"""
# Scale
# TRAIN_TOKENS = 100_000_000_000 #2**35


def read_sample_int16(stream, sample_size, index):
    byte_data = stream[2 * index * sample_size : 2 * (index + 1) * sample_size]
    return np.array(array.array("H", byte_data), dtype=np.int16)


class MMapDataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, sample_size):
        self.sample_size = sample_size
        self.path = path

        self.tokenizer = tokenizer
        self.pad = tokenizer.pad_token_id
        self.mm = None
        self.len = len(self)

    def __getitem__(self, index):
        # Hash index with GPU rank to ensure
        # A different data batch per device
        # Does not handle collisions
        rank = torch.distributed.get_rank()
        index = hash(str(index) + str(rank)) % self.len

        if self.mm is None:
            with open(self.path, "r") as f:
                self.mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
        return read_sample_int16(self.mm, self.sample_size, index)

    def __len__(self):
        with open(self.path, "r") as f:
            mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
            size = mm.size() / self.sample_size / 2
            assert size == int(size)
            return int(size)


class P3Stream(MMapDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        input_ids, labels = np.split(item, 2)

        decoder_input_ids = t5_collator.shift_tokens_right(
            labels.reshape(1, -1), self.pad, 0
        ).ravel()
        out = {
            "input_ids": input_ids,
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
        }

        out["input_ids"] = np.where(
            out["input_ids"] == -100, self.pad, out["input_ids"]
        )
        out["attention_mask"] = np.where(
            out["input_ids"] == self.pad, 0, out["input_ids"]
        )

        for key, val in out.items():
            out[key] = torch.LongTensor(val)
        return out


class T50Data(pl.LightningDataModule):
    def __init__(self, tokenizer, p3_batch, pile_batch):
        super().__init__()
        self.tokenizer = tokenizer
        self.p3_batch = p3_batch
        self.pile_batch = pile_batch

    def setup(self, stage=None):
        prefix = "data"
        if "KUBERNETES_SERVICE_HOST" in os.environ:
            prefix = "/mnt"

        self.p3_train = P3Stream(
            prefix + "/p3-stream/training", self.tokenizer, 2 * SAMPLE_SIZE
        )
        self.p3_val = P3Stream(
            prefix + "/p3-stream/valid", self.tokenizer, 2 * SAMPLE_SIZE
        )
        """
        self.p3_train = P3Stream(prefix + '/p3-stream/training', self.tokenizer, 2*consts.SAMPLE_SIZE)
        self.p3_val = P3Stream(prefix + '/p3-stream/valid', self.tokenizer, 2*consts.SAMPLE_SIZE)
        """

    def train_dataloader(self):
        kwargs = {"shuffle": True, "drop_last": True}
        cores = os.cpu_count()

        assert self.pile_batch == 0
        return DataLoader(
            self.p3_train, batch_size=self.p3_batch, num_workers=cores, **kwargs
        )

    def val_dataloader(self):
        kwargs = {
            "batch_size": self.pile_batch + self.p3_batch,
            "num_workers": os.cpu_count() // 2,
            "drop_last": True,
        }

        # Always validate on both datasets
        return DataLoader(self.p3_val, **kwargs)
