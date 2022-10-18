#!/bin/bash
NAME='tmp_new_alibi' #useful to always keep test runs in the same directory to ensure equality
MAX_STEPS=10000000
INTERVAL=1000000
SAVE_INTERVAL=1000000
CKPT='' #/mnt_out/colehawk/easel/tmp_1/08_01_16_43/epoch\=0-step\=40-validation_loss\=11.9797_training_loss_step\=11.9844.ckpt'
LIMIT_VAL_BATCHES=10 
#Usually testing on a g4

python3 pretrain_main.py \
    trainer.max_steps=$MAX_STEPS \
    data.new_datamodule=1 \
    data=t5_prod_drop_2 \
    model.ckpt_path=$CKPT \
    callback.save_every_n_train_steps=${SAVE_INTERVAL} \
    trainer.val_check_interval=${SAVE_INTERVAL} \
    trainer.limit_val_batches=${LIMIT_VAL_BATCHES} \
    run_name=${NAME} \
    trainer.log_every_n_steps=${INTERVAL}  \
    trainer.num_nodes=1 \
    optimizer.micro_batch_size=2 \
    optimizer.total_batch_size=8 \
    optimizer.warmup_steps=50 \
    model=20B \
    model.positional_embedding=alibi \
    trainer.num_sanity_val_steps=0 \
    trainer.precision=bf16 \
    deepspeed_path=config/deepspeed/bf16_zero2_shard24.json 
