#!/bin/bash
#NAME='300B_t5'
#NAME='200B_t5'
NAME='11B_t5'
STATE_DICT_PATH="package/tmp/${NAME}/model_fp32_state_dict"

DATA='t5_reddit'

python3 pretrain_main.py \
    data=$DATA \
    validate_only=1 \
    data.new_datamodule=1 \
    trainer.limit_val_batches=100 \
    run_name=$NAME \
    trainer.num_nodes=1 \
    optimizer.micro_batch_size=2\
    optimizer.total_batch_size=16 \
    model=11B\
    model.positional_embedding=alibi \
    deepspeed_path=config/deepspeed/zero2d.json \
    ++model.load_method=state_dict \
    ++model.state_dict_path=$STATE_DICT_PATH \
    ++save_automodel_from_state_dict=1\
 #$CKPT/checkpoint/mp_rank_00_model_states.pt\
    #model.ckpt_path=$CKPT \
