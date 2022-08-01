"""
Stream in the pile, filter, then write arrow file
"""
import argparse
import os
import io
import random
import json
import time
import multiprocessing
from multiprocessing import Pool
import pyarrow as pa
from smart_open import open  # use to read s3
import zstandard as zstd


def read_and_preprocess(file_name, filter_list):
    """
    read the file, filter out samples from filter_list, extract text, return as a list
    """
    print("Processing {}".format(file_name))
    all_text = []
    with open(file_name, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(fh)
        text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")
        for str_line in text_stream:
            line = json.loads(str_line)
            if line["meta"]["pile_set_name"] not in filter_list:
                text = line["text"].strip()
                if text == "":
                    continue
                assert isinstance(text, str)
                all_text.append(text)

    print("Finished preprocessing {} with {} lines".format(file_name, len(all_text)))
    return all_text


def write_arrow_file(all_text, output_file):
    """
    write all read and processed text into an arrow file
    all_text: list of strings, each element of the list will become one row of the arrow file
    """
    schema = pa.schema({"text": pa.large_string()})
    arr = pa.array(all_text, type=pa.large_string())
    try:
        schema = pa.schema({"text": pa.large_string()})
        arr = pa.array(all_text, type=pa.large_string())

        with pa.OSFile(output_file, "wb") as sink:
            with pa.ipc.new_stream(sink, schema=schema) as writer:
                batch = pa.record_batch([arr], schema=schema)
                writer.write(batch)

        # verify that chunk size is 1
        # if chunks>1 indexing is slow during training
        mmap = pa.memory_map(output_file)
        reload_for_test = pa.ipc.open_stream(mmap).read_all()
        assert (
            len(reload_for_test["text"].chunks) == 1
        ), "Indexing is much slower with more than 1 chunk"

    except Exception as e:
        print(e)
        return 0

    return len(arr)


def process_files(args):
    """
    read multiple .jsonl files in parallel
    preprocess to extract the text and write it into an arrow file
    """

    if args.input_file is not None:
        # next stage expects a list
        assert args.file_list is None, "Either provide file or filelist"
        file_list = [args.input_file]
    else:
        file_list = open(args.file_list, "r").read().splitlines()

    num_cores = min(multiprocessing.cpu_count(), len(file_list))

    with Pool(processes=num_cores) as pool:
        results = [
            pool.apply_async(
                read_and_preprocess,
                (
                    os.path.join(args.data_path_prefix, input_file),
                    args.filter_list,
                ),
            )
            for input_file in file_list
        ]

        all_text = []
        for res in results:
            all_text += res.get()

        print("Number of lines", len(all_text))
        print("Writing arrow file")
        length = write_arrow_file(all_text, args.output_file)
        print("Finished writing arrow file")
        print("number of samples in the file: {}".format(length))


def add_args(parser):
    """
    Adds logs arguments to the passed parser
    """
    parser.add_argument(
        "--data_path_prefix",
        default="s3://mstar-data/pile/",
        type=str,
        required=False,
        help="Path prefix of data files.",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="merged data file."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        required=False,
        help="Path to the single file to process and write.",
    )
    parser.add_argument(
        "--file_list",
        type=str,
        default=None,
        required=False,
        help="Path to the list of files to process and merge.",
    )
    parser.add_argument(
        "--filter_list",
        type=str,
        required=False,
        help="Path to the list of datasets to filter out while preprocessing.",
    )
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    start = time.time()

    assert (
        args.input_file is None or args.file_list is None
    ), "Either pass a list or a single file"

    # check if there is any file to list out datasets to filter out
    if os.path.isfile(args.filter_list):
        args.filter_list = open(args.filter_list, "r").read().splitlines()
    else:
        args.filter_list = []

    process_files(args)

    end = time.time()
    print("time to process: {} seconds".format(end - start))
