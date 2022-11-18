#!/bin/bash
NAME='tmp'
MAX_STEPS=10000
VAL_INTERVAL=10000
SAVE_INTERVAL=10000
CKPT='' 
LIMIT_VAL_BATCHES=10 
WARMUP_STEPS=100
MICRO_BATCH_SIZE=2 #per-device batch size

python3 pretrain_main.py \
    trainer.max_steps=$MAX_STEPS \
    data.new_datamodule=1 \
    data=t5_prod_drop_3 \
    model.ckpt_path=$CKPT \
    trainer.val_check_interval=${VAL_INTERVAL} \
    trainer.limit_val_batches=${LIMIT_VAL_BATCHES} \
    run_name=${NAME} \
    trainer.log_every_n_steps=10 \
    optimizer.micro_batch_size=${MICRO_BATCH_SIZE} \
    optimizer.warmup_steps=${WARMUP_STEPS} \
    model=tiny \
    trainer.precision=bf16 \
    trainer.num_sanity_val_steps=0 \
    deepspeed_path=config/deepspeed/bf16_zero2d4.json 
