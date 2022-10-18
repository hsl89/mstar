#!/bin/bash

MODEL=11B
MICRO=4
MACRO=128
NUM_NODES=4

python3 pretrain_main.py \
    model=$MODEL \
    optimizer.micro_batch_size=$MICRO \
    optimizer.total_batch_size=$MACRO \
    ++trainer.num_sanity_val_steps=0 \
    trainer.val_check_interval=10 \
    ++trainer.limit_val_batches=5 \
    ++deepspeed_path='config/deepspeed/zero2d.json' \
    trainer.num_nodes=$NUM_NODES \
#model.tie_word_embeddings=true\
#model.num_layers=2\
#model.num_decoder_layers=2\
