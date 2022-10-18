#!/bin/bash

MODEL=20B
MICRO=2
MACRO=64
NUM_NODES=4

python3 pretrain_main.py \
    model=$MODEL \
    optimizer.micro_batch_size=$MICRO \
    optimizer.total_batch_size=$MACRO \
    ++deepspeed_path='config/deepspeed/zero2d.json' \
    trainer.num_nodes=$NUM_NODES \
