#!/bin/bash
MAX_STEPS=10000
CKPT_INTERVAL=1000
CKPT='' 
LIMIT_VAL_BATCHES=10 
WARMUP_STEPS=100
MICRO_BATCH_SIZE=2 #per-device batch size

NUM_WORKERS=32
PREFETCH_FACTOR=4

NAME="tmp"

python3 pretrain_main.py \
    trainer.reload_dataloaders_every_n_epochs=0 \
    model=tiny  \
    data.num_workers=${NUM_WORKERS} \
    data.prefetch_factor=${PREFETCH_FACTOR} \
    trainer.log_every_n_steps=1\
    lightning.callbacks.progress_bar.refresh_rate=1\
    max_steps=$MAX_STEPS \
    model.ckpt_path=$CKPT \
    lightning.callbacks.checkpoint.every_n_train_steps=${CKPT_INTERVAL} \
    trainer.limit_val_batches=${LIMIT_VAL_BATCHES} \
    run_name=${NAME} \
    optimization.micro_batch_size=${MICRO_BATCH_SIZE} \
    optimization.scheduler.num_warmup_steps=${WARMUP_STEPS} \
    optimization.scheduler.scale_factor=${WARMUP_STEPS} \
    trainer.num_sanity_val_steps=0 \
    data=tmp \
    tokenizer.pretrained_model_name_or_path=/mnt/tokenizer/mstar-t5-20B-bedrock-stage_2_t5_600B_embed_fix-nfkc \
