#!/bin/bash

export REAL_WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES=0,1

python pretrain_main.py \
    model.arch_path=config/model/architecture/gpt2_124M_small.json \
    callback.save_every_n_steps=50 \
    trainer.val_check_interval=50 \
    model.ckpt_path='auto' \
    data.num_workers=1 \
    trainer.default_root_dir=/mnt_out/colehawk/dec_tmp/ \
    deepspeed_config=config/deepspeed/stage2d-bf16-shard2.json \
    data.training_dataset=/mnt/pretraining-data/package-01-06-23-v1/val.arrow \
    data.validation_dataset=/mnt/pretraining-data/package-01-06-23-v1/val.arrow \
    trainer.limit_val_batches=10 \
