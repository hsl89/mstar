#!/bin/bash
NAME='tmp_new_alibi' #useful to always keep test runs in the same directory to ensure equality
MAX_STEPS=200
INTERVAL=25
SAVE_INTERVAL=25
CKPT='' #/mnt_out/colehawk/easel/tmp_1/08_01_16_43/epoch\=0-step\=40-validation_loss\=11.9797_training_loss_step\=11.9844.ckpt' 

#Usually testing on a g4

python3 pretrain_main.py \
    trainer.max_steps=$MAX_STEPS \
    data.new_datamodule=1 \
    model.ckpt_path=$CKPT \
    callback.save_every_n_train_steps=${SAVE_INTERVAL} \
    trainer.val_check_interval=${SAVE_INTERVAL} \
    trainer.limit_val_batches=${INTERVAL} \
    run_name=${NAME} \
    trainer.log_every_n_steps=${INTERVAL}  \
    trainer.num_sanity_val_steps=0 \
    trainer.precision=bf16 \
    deepspeed_path=config/deepspeed/bf16_stage2.json \
    ++model.load_method=automodel \
    optimizer.micro_batch_size=2 \
    trainer.num_nodes=1 \
    optimizer.total_batch_size=16 \
    data=stage_2_clm \
    trainer.max_steps=100\
    optimizer.lr_scheduler_type=linear \
    optimizer.warmup_steps=100 \
    ++model.automodel_path=/mnt/colehawk/bedrock_prod_automodels/stage_1/1_9B/ \
    ++data.autotokenizer_path=/mnt/colehawk/bedrock_prod_automodels/tokenizer \

#data = stage_2_clm \
