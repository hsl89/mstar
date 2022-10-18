#!/bin/bash
NAME='tmp' #useful to always keep test runs in the same directory to ensure equality
MAX_STEPS=200
INTERVAL=10
SAVE_INTERVAL=10
CKPT=''

#Usually testing on a g4
python3 pretrain_main.py \
    trainer.max_steps=$MAX_STEPS \
    data.new_datamodule=1 \
    data=t5_reddit \
    model.ckpt_path=$CKPT \
    callback.save_every_n_train_steps=${SAVE_INTERVAL} \
    trainer.val_check_interval=${SAVE_INTERVAL} \
    trainer.limit_val_batches=${INTERVAL} \
    run_name=${NAME} \
    trainer.log_every_n_steps=${INTERVAL}  \
    trainer.num_nodes=2 \
    optimizer.micro_batch_size=2 \
    optimizer.total_batch_size=32 \
    optimizer.warmup_steps=50 \
    model.num_layers=2 \
    model.num_decoder_layers=2 \
    model.positional_embedding=alibi \
    trainer.num_sanity_val_steps=0 \
    deepspeed_path=config/deepspeed/zero2d.json \
