import argparse
import os
import numpy as np


def write_mmap_file(output_path, file_index, arr):
    fname = os.path.join(output_path, str(file_index) + '.mmap')
    arr_mmap = np.memmap(fname, dtype='uint32', mode='w+', shape=arr.shape)
    arr_mmap[:] = arr[:]
    return


def generate_index_files(args):

    # number of indexes to be generated in each file, except the last one.
    n_index = int(np.floor((args.end_index - args.start_index)/ args.num_files))
    start = args.start_index
    rng = np.random.default_rng(args.seed)
    arr = np.arange(start, args.end_index)
    rng.shuffle(arr)
    for i in range(args.num_files-1):
        arr_split = arr[start:start+n_index]
        start = start+n_index
        write_mmap_file(args.output_path, i, arr_split)

    # write the last index file
    arr_split = arr[start:]
    write_mmap_file(args.output_path, args.num_files-1, arr_split)


def add_args(parser):
    """
    Adds logs arguments to the passed parser
    """
    parser.add_argument("--num_files", default=8, type=int, required=False,
                        help="number of index files to be generated")
    parser.add_argument("--start_index", default=0, type=int, required=False,
                        help="start index of the generated files")
    parser.add_argument("--end_index", default=1000000, type=int, required=False,
                        help="end index of the generated files")
    parser.add_argument("--seed", default=1234, type=int, required=False,
                        help="random seed to keep the generated sequence reproducible")
    parser.add_argument("--output_path", default="/home/ubuntu/index_files/", type=str, required=False,
                        help="merged data file.")
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    args = parser.parse_args()

    generate_index_files(args)
