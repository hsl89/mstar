#!/bin/bash

MODEL=20B
MICRO=4
MACRO=1024
NUM_NODES=32

python3 pretrain_main.py \
    model=$MODEL \
    ++model.positional_embedding=t5 \
    optimizer.micro_batch_size=$MICRO \
    optimizer.total_batch_size=$MACRO \
    ++trainer.num_sanity_val_steps=0 \
    trainer.val_check_interval=50 \
    ++trainer.limit_val_batches=5 \
    ++deepspeed_path='config/deepspeed/zero2d.json' \
    trainer.num_nodes=$NUM_NODES \
#model.num_layers=2\
#model.num_decoder_layers=2\
