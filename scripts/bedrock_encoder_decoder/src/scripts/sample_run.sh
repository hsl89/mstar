#!/bin/bash
NAME='tmp_sqrt_decay'
MAX_STEPS=10000
VAL_INTERVAL=100
CKPT='' 
LIMIT_VAL_BATCHES=10 
WARMUP_STEPS=100
MICRO_BATCH_SIZE=2 #per-device batch size

python3 pretrain_main.py \
    max_steps=$MAX_STEPS \
    model.ckpt_path=$CKPT \
    trainer.val_check_interval=${VAL_INTERVAL} \
    lightning.callbacks.checkpoint.every_n_train_steps=${VAL_INTERVAL} \
    trainer.limit_val_batches=${LIMIT_VAL_BATCHES} \
    run_name=${NAME} \
    optimization.micro_batch_size=${MICRO_BATCH_SIZE} \
    optimization.scheduler.num_warmup_steps=${WARMUP_STEPS} \
    optimization.scheduler.scale_factor=${WARMUP_STEPS} \
    trainer.num_sanity_val_steps=0 \
