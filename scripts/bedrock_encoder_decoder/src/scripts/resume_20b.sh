#!/bin/bash
NAME='tmp_alibi' #useful to always keep test runs in the same directory to ensure equality
MAX_STEPS=100
INTERVAL=10
SAVE_INTERVAL=10
CKPT='/mnt_out/colehawk/easel/tmp_alibi/09_07_17_59/epoch\=0-step\=30-validation_loss\=8.6246_training_loss_step\=8.0586.ckpt/' 

#Usually testing on a g4

python3 pretrain_main.py \
    trainer.max_steps=$MAX_STEPS \
    data.new_datamodule=1 \
    model=20B \
    data=t5_reddit \
    model.ckpt_path=$CKPT \
    run_name=${NAME} \
    optimizer.micro_batch_size=8 \
    trainer.num_nodes=1 \
    optimizer.total_batch_size=64 \
    trainer.num_sanity_val_steps=0 \
    optimizer.base_learning_rate=0.0002 \
    deepspeed_path=config/deepspeed/bf16_zero2d.json \
    trainer.precision=bf16 \
    model.num_layers=2 \
    model.num_decoder_layers=2 \
    callback.save_every_n_train_steps=${SAVE_INTERVAL} \
    trainer.val_check_interval=${SAVE_INTERVAL} \
    trainer.limit_val_batches=${INTERVAL} \
    trainer.log_every_n_steps=${INTERVAL}  \
    optimizer.warmup_steps=50 \
    #model.positional_embedding=t5 \
