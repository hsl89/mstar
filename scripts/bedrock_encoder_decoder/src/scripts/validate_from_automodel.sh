#!/bin/bash
#NAME='300B_t5'
#NAME='200B_t5'
NAME='tmp_val'

AUTOMODEL_PATH='/mnt/colehawk/bedrock_prod_automodels/alexatm_mix/hf_automodel/'
AUTOTOKENIZER_PATH='/mnt/colehawk/bedrock_prod_automodels/alexatm_mix/tokenizer'
DATA='mix_high_mask_reddit'

AUTOMODEL_PATH='/mnt/colehawk/bedrock_prod_automodels/alexatm_mix/hf_automodel/'
#AUTOMODEL_PATH='/mnt/colehawk/bedrock_prod_automodels/alexatm_mix/hf_automodel/'
AUTOTOKENIZER_PATH='/mnt/colehawk/bedrock_prod_automodels/alexatm_mix/tokenizer'
#AUTOTOKENIZER_PATH='/mnt/colehawk/bedrock_prod_automodels/alexatm_mix/tokenizer'
DATA='mix_high_mask_reddit'

AUTOMODEL_PATH='t5_tmp'
#AUTOMODEL_PATH='/mnt/colehawk/bedrock_prod_automodels/t5_token_count/300B'
AUTOTOKENIZER_PATH='t5_tmp/tokenizer'
#AUTOTOKENIZER_PATH='/mnt/colehawk/bedrock_prod_automodels/t5_token_count/300B/tokenizer'
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
    deepspeed_path=config/deepspeed/stage2.json \
    ++model.load_method=automodel \
    ++model.automodel_path=$AUTOMODEL_PATH\
    ++data.autotokenizer_path=$AUTOTOKENIZER_PATH\
 #$CKPT/checkpoint/mp_rank_00_model_states.pt\
    #model.ckpt_path=$CKPT \
