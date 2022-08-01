#!/bin/bash

S3_BUCKET='s3://mstar-data/pile'
MOUNT_DIR='/mnt/pile_mnt'
FOLDER='' #pile_no_youtube

#process val file
python pile_data_preprocess.py \
    --data_path_prefix $S3_BUCKET \
    --filter_list configs/subsets_to_exclude.txt \
    --output_file ${MOUNT_DIR}/val.arrow \
    --input_file val.jsonl.zst 

#process test file
python pile_data_preprocess.py \
    --data_path_prefix $S3_BUCKET \
    --filter_list configs/subsets_to_exclude.txt \
    --output_file ${MOUNT_DIR}/test.arrow \
    --input_file test.jsonl.zst 

#process all train files
python pile_data_preprocess.py \
    --data_path_prefix ${S3_BUCKET}/train \
    --filter_list configs/subsets_to_exclude.txt \
    --file_list configs/full_file_list.txt \
    --output_file ${MOUNT_DIR}/training.arrow 
