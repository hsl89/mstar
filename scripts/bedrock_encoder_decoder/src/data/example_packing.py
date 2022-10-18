"""
From preprocessed arrow file, pack examples to  a target sequence length
"""
import argparse
import os
import multiprocessing
from multiprocessing import Pool
import numpy as np
import pyarrow as pa
import json
import pyarrow.dataset as ds
import time
from tqdm import tqdm
from transformers import AutoTokenizer
import datasets
from itertools import repeat
from collections import deque
import warnings
import random
import pickle

CHUNKSIZE = 1000  # how many examples to group together
MAX_CONCAT_SIZE = (
    1e8  # max string_length in chars, stop building string after exceedence
)
TOKENIZER = "t5-base"


def example_pack(input_tuple, separator_token=None):
    """
    Pack examples up to max_sequence_length
    Input examples are concatenated into long chunks
    separated by double EOS token from tokenizer or separator_token. 
    Then examples are tokenized and split to reach max_seq_len
    Best to use the pretraining tokenizer here
    For T5-style example packing, want (1+masking_ratio)*desired_downstream_sequence_length due to denoising objective
    """
    # os.environ["TOKENIZERS_PARALLELISM"] = "0"
    args, idx, tokenizer = input_tuple

    # all processes must shuffle with the same seed
    to_pack = datasets.arrow_dataset.Dataset.from_file(args.data_file).shuffle(
        seed=args.seed
    )

    examples_list = to_pack.select(
        list(range(idx, min(idx + CHUNKSIZE, len(to_pack))))
    )["text"]
    # avoid tokenizer max_length error
    tokenizer.model_max_length = MAX_CONCAT_SIZE

    separator_token = tokenizer.eos_token

    packed_examples = []
    to_pack = deque(examples_list)

    # iterate over queued examples and pack
    while to_pack:
        to_split = ""
        while len(to_split) < MAX_CONCAT_SIZE and len(to_pack) > 0:
            # pack on next example
            to_split += to_pack.pop()
            # add separator token
            to_split += separator_token

        tokenized_to_split = tokenizer.encode(to_split, truncation=False)

        # chunk into max_seq_length-sized chunks, drop last example
        tokenized_packed_batch = [
            tokenized_to_split[i : i + args.example_pack_length]
            for i in range(0, len(tokenized_to_split), args.example_pack_length)
        ]

        # drop last example which is likely to be short
        tokenized_packed_batch.pop(-1)

        # decode back to text
        packed_batch = [tokenizer.decode(x) for x in tokenized_packed_batch]

        packed_examples += packed_batch

    random.shuffle(packed_examples)

    return packed_examples


def write_arrow_file(all_text, output_file):
    """
    write all read and processed text into an arrow file
    """
    schema = pa.schema({"text": pa.large_string()})
    arr = pa.array(all_text, type=pa.large_string())

    with pa.OSFile(output_file, "wb") as sink:
        with pa.ipc.new_stream(sink, schema=schema) as writer:
            batch = pa.record_batch([arr], schema=schema)
            writer.write(batch)
    print("Finished writing {}".format(output_file))

    return len(arr)


def process(args):
    """
    process arrow file in chunks
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    # get CHUNK LENGTH
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER
    )  # workaround to avoid recreation

    mmap = pa.memory_map(args.data_file)
    to_pack = pa.ipc.open_stream(mmap).read_all()

    assert (
        len(to_pack["text"].chunks) == 1
    ), "Indexing is much slower with more than 1 chunk"

    num_idx = len(to_pack)
    chunk_idxs = [(args, idx, tokenizer) for idx in list(range(0, num_idx, CHUNKSIZE))]

    num_cores = (
        os.cpu_count() if os.cpu_count() <= len(chunk_idxs) else len(chunk_idxs)
    )  # may hang with too many workers
    print(
        "Packing {} examples to sequence length {} with {} cores".format(
            num_idx, args.example_pack_length, num_cores
        )
    )

    with Pool(processes=num_cores) as pool:
        tmp = tqdm(pool.imap(example_pack, chunk_idxs))
        # all_text = [y for x in tmp for y in x]

        all_text = [y for x in tmp for y in x]

    print("Packing complete, writing to arrow")
    output_file = args.data_file.split(".arrow")[
        0
    ] + "_packed_chunksize_{}.arrow".format(args.example_pack_length)
    length = write_arrow_file(all_text, output_file)
    print("number of samples in the file: {}".format(length))


def add_args(parser):
    """
    Adds logs arguments to the passed parser
    """
    parser.add_argument(
        "--data_file",
        default="/mnt/pile_mnt/pile/train/training.arrow",
        type=str,
        required=False,
        help="Arrow data table with examples to pack.",
    )
    parser.add_argument(
        "--example_pack_length",
        type=int,
        required=True,
        help="Specifies the size to pack examples to. For instance 4096 means examples are concatenated and then chunked into size-4096 tokens",
    )
    parser.add_argument("--seed", default=1234, type=int)
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    start = time.time()

    # packs chunks from arrow table and write to new table
    process(args)

    end = time.time()
    print("time to process: {} seconds".format(end - start))
