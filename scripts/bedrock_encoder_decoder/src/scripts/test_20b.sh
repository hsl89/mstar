#!/bin/bash
NAME='test_fsx' #useful to always keep test runs in the same directory to ensure equality
MAX_STEPS=100
INTERVAL=3
SAVE_INTERVAL=3
CKPT='' #/mnt_out/colehawk/easel/tmp_1/08_01_16_43/epoch\=0-step\=40-validation_loss\=11.9797_training_loss_step\=11.9844.ckpt' 

#Usually testing on a g4

python3 pretrain_main.py \
    trainer.max_steps=$MAX_STEPS \
    data.new_datamodule=1 \
    model=20B \
    data=t5_prod \
    model.ckpt_path=$CKPT \
    run_name=${NAME} \
    optimizer.micro_batch_size=2 \
    trainer.num_nodes=2 \
    optimizer.total_batch_size=32 \
    trainer.num_sanity_val_steps=0 \
    optimizer.base_learning_rate=0.0002 \
    deepspeed_path=config/deepspeed/bf16_zero2d.json \
    trainer.precision=bf16 \
    callback.save_every_n_train_steps=${SAVE_INTERVAL} \
    trainer.val_check_interval=${SAVE_INTERVAL} \
    trainer.limit_val_batches=${INTERVAL} \
    trainer.log_every_n_steps=${INTERVAL}  \
    optimizer.warmup_steps=50 \
    #model.positional_embedding=t5 \
