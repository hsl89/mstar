#!/bin/bash
NAME='t5_alibi_val'
#/mnt_out/colehawk/easel/1_9B_t5_prod_candidate/09_01_10_58/epoch\=0-step\=85000-validation_loss\=1.2147_training_loss_step\=1.0209.ckpt/'
CKPT='/mnt_out/colehawk/easel/1_9B_t5_prod_candidate/09_06_15_02/epoch\=1-step\=300000-validation_loss\=1.0936_training_loss_step\=0.9556.ckpt/'
STATE_DICT_PATH=''

DATA='t5_reddit'

python3 pretrain_main.py \
    data=$DATA \
    validate_only=1 \
    data.new_datamodule=1 \
    trainer.limit_val_batches=20 \
    run_name=$NAME \
    trainer.num_nodes=1 \
    optimizer.micro_batch_size=2\
    optimizer.total_batch_size=16 \
    model.positional_embedding=alibi \
    ++model.load_method=state_dict \
    ++model.state_dict_path=$CKPT/checkpoint/mp_rank_00_model_states.pt\
    #model.ckpt_path=$CKPT \
