#!/bin/bash
NAME='tmp' #useful to always keep test runs in the same directory to ensure equality
MAX_STEPS=10000000
INTERVAL=25
SAVE_INTERVAL=50
CKPT='' #/mnt_out/colehawk/easel/tmp_1/08_01_16_43/epoch\=0-step\=40-validation_loss\=11.9797_training_loss_step\=11.9844.ckpt'
LIMIT_VAL_BATCHES=10 
#Usually testing on a g4

python3 pretrain_main.py \
    trainer.max_steps=$MAX_STEPS \
    data.new_datamodule=1 \
    data=t5_prod_drop_3 \
    model.ckpt_path=$CKPT \
    callback.save_every_n_train_steps=${SAVE_INTERVAL} \
    trainer.val_check_interval=${INTERVAL} \
    trainer.limit_val_batches=${LIMIT_VAL_BATCHES} \
    run_name=${NAME} \
    trainer.log_every_n_steps=${INTERVAL}  \
    trainer.num_nodes=1 \
    optimizer.micro_batch_size=2 \
    optimizer.total_batch_size=8 \
    optimizer.warmup_steps=100 \
    model=1_9B \
    ++model.load_method=automodel \
    ++model.automodel_path=/mnt/colehawk/bedrock_prod_automodels/stage_1/1_9B/ \
    ++data.autotokenizer_path=/mnt/colehawk/bedrock_prod_automodels/tokenizer/ \
    model.positional_embedding=alibi \
    trainer.num_sanity_val_steps=0 \
    trainer.precision=bf16 \
    deepspeed_path=config/deepspeed/bf16_zero2d4.json 
