#!/bin/bash
NUM_GPUS=256
OUTPUT_DIR="/tmp/indices/${NUM_GPUS}_gpus"
SEED=1234

mkdir -p $OUTPUT_DIR/train
mkdir -p $OUTPUT_DIR/val
mkdir -p $OUTPUT_DIR/test

#train
python generate_index_files.py --num_files $NUM_GPUS --start_index 0 --end_index 142761690 --seed $SEED --output_path $OUTPUT_DIR/train

#test

python generate_index_files.py --num_files $NUM_GPUS --start_index 0 --end_index 142640 --seed $SEED --output_path $OUTPUT_DIR/test


#val
python generate_index_files.py --num_files $NUM_GPUS --start_index 0 --end_index 146127 --seed $SEED --output_path $OUTPUT_DIR/val
