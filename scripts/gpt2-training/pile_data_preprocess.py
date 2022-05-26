import argparse
import os
import multiprocessing
from multiprocessing import Pool
import numpy as np
import pyarrow as pa
from pyarrow import feather
import json
import pyarrow.dataset as ds
import time
from tqdm import tqdm

def read_and_preprocess(file_name, filter_list):
    '''
    read the file, filter out samples from filter_list, extract text, return as a list
    '''
    all_text = []
    with open(file_name) as f:
        for line in tqdm(f):
            if json.loads(line)['meta']['pile_set_name'] not in filter_list:
                text = json.loads(line)['text'].strip()
                if text == "":
                    continue
                assert isinstance(text, str)
                all_text.append(text)
    print("Finished preprocessing {}".format(file_name))
    return all_text

def write_arrow_file(all_text, output_file):
    '''
    write all read and processed text into an arrow file
    '''
    try:
        schema = pa.schema({'text': pa.large_string()})
        arr = pa.array(all_text, type=pa.large_string())
        with pa.OSFile(output_file, 'wb') as sink:
            with pa.ipc.new_stream(sink, schema=schema) as writer:
                batch = pa.record_batch([arr], schema=schema)
                writer.write(batch)
        print("Finished writing {}".format(output_file))

    except Exception as e:
        print(e)
        return 0
    return len(arr)

def process_single_file(args):
    '''
    read a single file in .jsonl format, preprocess to extract the text and write it into an arrow file
    '''
    all_text = read_and_preprocess(args.input_file, args.filter_list)
    length = write_arrow_file(all_text, args.output_file)
    print('number of samples in the file: {}'.format(length))
    return

def process_multiple_files(args):
    '''
    read multiple .jsonl files in parallel, preprocess to extract the text and write it into an arrow file
    '''
    file_list = open(args.file_list, 'r').read().splitlines()
    num_cores = multiprocessing.cpu_count()
    num_cores = num_cores if num_cores <= len(file_list) else len(file_list)
    with Pool(processes=num_cores) as pool:
        # launching multiple evaluations asynchronously *may* use more processes
        results = [pool.apply_async(read_and_preprocess, (os.path.join(args.data_path_prefix, input_file), \
                                                          args.filter_list,)) for input_file in file_list]

        all_text = []
        for res in results:
            all_text += res.get()
        print(len(all_text))

        length = write_arrow_file(all_text, args.output_file)
        print('number of samples in the file: {}'.format(length))
    return


def add_args(parser):
    """
    Adds logs arguments to the passed parser
    """
    parser.add_argument("--data_path_prefix", default="/home/ubuntu/data", type=str, required=False,
                        help="Path prefix of training and validation data.")
    parser.add_argument("--output", default="/hdd1/s3/cc100.txt.xz/", type=str, required=False,
                        help="Path prefix of training and validation data.")
    parser.add_argument("--output_file", default="/hdd1/s3/cc100.txt.xz/training.arrow", type=str, required=False,
                        help="merged data file.")
    parser.add_argument("--single_file", action='store_true', required=False,
                        help="use when processing single file, not a list of files")
    parser.add_argument("--input_file", default="/home/ubuntu/pile/test.jsonl", type=str, required=False,
                        help="Path to the file to process and write.")
    parser.add_argument("--file_list", default="/home/ubuntu/train_list.txt", type=str, required=False,
                        help="Path to the list of files to process and merge.")
    parser.add_argument("--filter_list", default="/home/ubuntu/filter_list.txt", type=str, required=False,
                        help="Path to the list of datasets to filter out while preprocessing.")
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()
    start = time.time()
    # check if there is any file to list out datasets to filter out
    if os.path.isfile(args.filter_list):
        args.filter_list = open(args.filter_list, 'r').read().splitlines()
    else:
        args.filter_list = []

    if args.single_file:
        # the case with single file, e.g. test or validation file
        process_single_file(args)
    else:
        process_multiple_files(args)

    end = time.time()
    print('time to process: {} seconds'.format(end - start))
