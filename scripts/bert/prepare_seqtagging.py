"""Pretraining on code"""
import math
import tqdm
import argparse
import os
import pathlib
import random
import multiprocessing

import mstar

# from transformers import AutoTokenizer
from smart_open import open
import gluonnlp as nlp
import numpy as np
import pyarrow as pa
import pyarrow.feather
import pyarrow.compute


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )

    # # Model
    group = parser.add_argument_group("Model")
    group.add_argument(
        "--model-name",
        type=str,
        default="google_en_uncased_bert_base",
        choices=mstar.models.bert.bert_cfg_reg.list_keys(),
        help="Name of the model configuration.",
    )

    # Input
    group = parser.add_argument_group("Input")
    group.add_argument(
        "--input-directory",
        type=pathlib.Path,
        help="Path to dirctory containing the data.",
    )
    group.add_argument(
        "--label-vocab",
        type=pathlib.Path,
        help="Path to file containing labels.",
    )
    group = parser.add_argument_group("Output")
    parser.add_argument("--output-directory", required=True, type=pathlib.Path)

    # data pre-processing
    group = parser.add_argument_group("Data pre-processing")
    group.add_argument(
        "--dupe-factor",
        type=int,
        default=5,
        help="Number of times to duplicate the \
            input data (with different masks).",
    )
    group.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum input sequence length.",
    )
    group.add_argument(
        "--short-seq-prob",
        type=float,
        default=0.1,
        help="The probability of producing sequences \
             shorter than max_seq_length.",
    )
    group.add_argument(
        "--masked-lm-prob",
        type=float,
        default=0.15,
        help="Probability for masks.",
    )
    group.add_argument(
        "--max-predictions-per-seq",
        type=int,
        default=80,
        help="Maximum number of predictions per sequence.",
    )

    # Computation and communication
    group = parser.add_argument_group("Computation")
    group.add_argument(
        "--processes",
        type=int,
        default=os.cpu_count(),
        help="Number of processes in process pool.",
    )
    group.add_argument("--seed", type=int, default=100, help="Random seed")

    # Misc
    group = parser.add_argument_group("Misc")
    group.add_argument("--logging-steps", type=int, default=10)

    args = parser.parse_args()

    return args


def set_seed(seed):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)


def create_masked_lm_predictions(
    *, args, tokens, cls_token_id, sep_token_id, mask_token_id, non_special_ids
):
    """Creates the predictions for the masked LM objective."""
    cand_indexes = [
        i
        for i, tok in enumerate(tokens)
        if tok not in (cls_token_id, sep_token_id)
    ]
    output_tokens = list(tokens)
    random.shuffle(cand_indexes)
    num_to_predict = min(
        args.max_predictions_per_seq,
        max(1, int(round(len(tokens) * args.masked_lm_prob))),
    )
    mlm_positions = []
    mlm_labels = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(mlm_positions) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)
        masked_token = None

        # 80% of the time, replace with [MASK]
        if random.random() < 0.8:
            masked_token = mask_token_id
        else:
            # 10% of the time, keep original
            if random.random() < 0.5:
                masked_token = tokens[index]
                # 10% of the time, replace with random word
            else:
                masked_token = random.choice(non_special_ids)

        output_tokens[index] = masked_token
        mlm_positions.append(index)
        mlm_labels.append(tokens[index])
    assert len(mlm_positions) <= num_to_predict
    assert len(mlm_positions) == len(mlm_labels)
    return output_tokens, mlm_positions, mlm_labels


def process_file(path_pair):
    # Fetch the filenames from the input
    input_file, output_file = path_pair
    # Retrieve process-local state
    tokenizer = process_file.tokenizer
    schema = process_file.schema
    args = process_file.args
    vocab = process_file.vocab
    non_special_ids = process_file.non_special_ids

    # Process file
    pa_batches = []  # List of PyArrow RecordBatches
    buffers = [
        [] for _ in range(len(schema))
    ]  # Arrays in same order as schema
    with open(input_file, "r", encoding="utf-8") as f:
        document_lines = f.readlines()

    # Find out document boundaries
    document_separator = "-DOCSTART-"
    # Find wherever documents start, removing the separator and newline
    document_boundary_indices = [
        i
        for i, x in enumerate(document_lines)
        if x.startswith(document_separator)
    ]
    document_segmented_data = [
        document_lines[i + 2: j]
        for i, j in zip(
            document_boundary_indices,
            document_boundary_indices[1:] + [len(document_lines)],
        )
    ]

    print(
        "Found {} document splits in input file {}".format(
            len(document_boundary_indices), len(input_file)
        )
    )

    for document_data in tqdm.tqdm(document_segmented_data):
        # According to the original tensorflow implementation: We *sometimes*
        # (i.e., short_seq_prob == 0.1, 10% of the time) want to use shorter
        # sequences to minimize the mismatch between pre-training and
        # fine-tuning.
        target_seq_length = args.max_seq_length - 2  # [CLS] ... [SEP]
        if random.random() < args.short_seq_prob:
            target_seq_length = random.randint(2, target_seq_length)

        # Split the data into columns, pick the first two columns,
        # and remove blank lines
        document_data_cleaned = list(
            filter(lambda x: len(x) > 0, map(str.split, document_data))
        )
        # Get only the tokens and the labels
        text, seqtagging_labels = zip(
            *[(x[0], x[-1]) for x in document_data_cleaned]
        )
        text, seqtagging_labels = list(text), list(seqtagging_labels)
        # Encode the seqtagging_labels
        seqtagging_labels = list(map(args.label_vocab.get, seqtagging_labels))

        # Tokenize
        toks = tokenizer.encode(text, int)
        # Repeat the sequence tagging labels for each wordpiece
        repeated_seqtagging_labels = [
            [seqtagging_labels[i]] * len(toks[i])
            for i in range(len(seqtagging_labels))
        ]
        all_toks, all_seqtagging_labels = sum(toks, []), sum(
            repeated_seqtagging_labels, []
        )
        assert len(all_toks) == len(
            all_seqtagging_labels
        ), "Mismatch in number of labels and tokens"

        # Iterate over target_seq_length chunks
        for i in range(math.ceil(len(all_toks) / target_seq_length)):
            start_index, end_index = i * target_seq_length, min(
                len(all_toks), (i + 1) * target_seq_length
            )
            final_toks, final_seqtagging_labels = (
                all_toks[start_index:end_index],
                all_seqtagging_labels[start_index:end_index],
            )
            mlm_toks, mlmpositions, mlmlabels = create_masked_lm_predictions(
                args=args,
                tokens=final_toks,
                cls_token_id=vocab.cls_id,
                sep_token_id=vocab.sep_id,
                mask_token_id=vocab.mask_id,
                non_special_ids=non_special_ids,
            )

            # Arrays in same order as schema
            buffers[0].append(mlm_toks)
            buffers[1].append(len(mlm_toks))
            buffers[2].append(mlmpositions)
            buffers[3].append(mlmlabels)
            buffers[4].append(final_seqtagging_labels)

            if len(buffers[0]) >= 10000000:
                batch = pa.RecordBatch.from_arrays(buffers, schema=schema)
                pa_batches.append(batch)
                # Reset the buffers
                buffers = [[] for _ in range(len(schema))]

    # Add the last set of data points left in the buffer
    if len(buffers[0]):
        batch = pa.RecordBatch.from_arrays(buffers, schema=schema)
        pa_batches.append(batch)
        buffers = [[] for _ in range(len(schema))]

    batch_tbl = pa.Table.from_batches(pa_batches, schema=schema)
    pa.feather.write_feather(batch_tbl, output_file + ".feather")


def process_file_paths(args):
    # Get the input directory structure
    current_directory = os.getcwd()
    os.chdir(args.input_directory)
    dirs, input_file_paths, output_file_paths = [], [], []
    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith(".txt"):
                dirs.append(root)
                input_file_paths.append(
                    os.path.join(args.input_directory, root, file)
                )
                output_file_paths.append(
                    os.path.join(
                        args.output_directory, root, file.replace(".txt", "")
                    )
                )
    os.chdir(current_directory)

    # Create output directory structure
    for directory in set(dirs):
        os.makedirs(
            os.path.join(args.output_directory, directory), exist_ok=True
        )

    print(
        f"Processing {len(input_file_paths)} input files \
            {args.dupe_factor} times."
    )

    # Duplicate input files based on dupe-factor
    all_input_file_paths = sum(
        (input_file_paths for _ in range(args.dupe_factor)), []
    )
    all_output_file_paths = sum(
        (
            [x + str(i) for x in output_file_paths]
            for i in range(args.dupe_factor)
        ),
        [],
    )

    assert len(all_input_file_paths) == len(all_output_file_paths)

    args.path_pairs = list(zip(all_input_file_paths, all_output_file_paths))


def main():
    args = parse_args()
    set_seed(args.seed)
    process_file_paths(args)

    # Load the label vocabulary
    assert os.path.exists(
        args.label_vocab
    ), "Label Vocabulary could not be found"

    with open(args.label_vocab, "r", encoding="utf-8") as f:
        labels = f.readlines()
    label_vocab = {label.rstrip(): i for i, label in enumerate(labels)}
    args.label_vocab = label_vocab

    def _initializer(function):
        """Initialize state of each process in multiprocessing pool.

        The process local state is stored as an attribute of the function
        object, which is specified in Pool(..., initargs=(function, )) and by
        convention refers to the function executed during map.

        """
        _, tokenizer, _, _ = nlp.models.bert.get_pretrained_bert(
            args.model_name, load_backbone=False, load_mlm=False
        )
        # tokenizer=AutoTokenizer.from_pretrained('facebook/mbart-large-cc25')
        function.tokenizer = tokenizer
        function.args = args
        function.vocab = tokenizer.vocab
        function.non_special_ids = tokenizer.vocab[
            tokenizer.vocab.non_special_tokens
        ]

        tok_type = (
            pa.uint16()
            if len(tokenizer.vocab) <= np.iinfo(np.uint16).max
            else pa.uint32()
        )
        assert len(tokenizer.vocab) <= np.iinfo(np.uint32).max
        length_type = pa.uint16()
        assert args.max_seq_length * 2 <= np.iinfo(np.uint16).max

        # pa.large_list instead of pa.list_ to use 64bit offsets
        # See https://issues.apache.org/jira/browse/ARROW-9773
        schema = pa.schema(
            {
                "tokens": pa.large_list(tok_type),
                "validlength": length_type,
                "mlmpositions1": pa.large_list(length_type),
                "mlmlabels": pa.large_list(tok_type),
                "seqtagging_labels": pa.large_list(tok_type),
            }
        )
        function.schema = schema

    if args.processes:
        with multiprocessing.Pool(
            initializer=_initializer,
            initargs=(process_file,),
            processes=args.processes,
        ) as pool:
            # pool.map(process_file, args.inputs) with tqdm progress bar
            with tqdm.tqdm(total=len(args.path_pairs)) as pbar:
                for i, _ in enumerate(
                    pool.imap_unordered(process_file, args.path_pairs)
                ):
                    pbar.update()
    else:
        _initializer(process_file)
        for _ in tqdm.tqdm(
            map(process_file, args.path_pairs), total=len(args.path_pairs)
        ):
            pass


if __name__ == "__main__":
    main()
