#!/bin/bash
NAME='test_automodel_resume'
MAX_STEPS=10000000
INTERVAL=50
SAVE_INTERVAL=50
CKPT='' #/mnt_out/colehawk/easel/tmp_1/08_01_16_43/epoch\=0-step\=40-validation_loss\=11.9797_training_loss_step\=11.9844.ckpt'
LIMIT_VAL_BATCHES=50 
#Usually testing on a g4

python3 /usr/local/src/mstar/scripts/bedrock_encoder_decoder/src/pretrain_main.py \
    trainer.max_steps=$MAX_STEPS \
    data.new_datamodule=1 \
    data=drop_2_stage_2_bart \
    model.ckpt_path=$CKPT \
    callback.save_every_n_train_steps=${SAVE_INTERVAL} \
    trainer.val_check_interval=${SAVE_INTERVAL} \
    trainer.limit_val_batches=${LIMIT_VAL_BATCHES} \
    run_name=${NAME} \
    trainer.log_every_n_steps=${INTERVAL}  \
    trainer.num_nodes=1 \
    optimizer.micro_batch_size=2 \
    optimizer.total_batch_size=16 \
    optimizer.warmup_steps=1000 \
    optimizer.base_learning_rate=0.0005 \
    model=11B \
    model.positional_embedding=alibi \
    trainer.num_sanity_val_steps=0 \
    trainer.precision=bf16 \
    deepspeed_path=config/deepspeed/bf16_zero2d.json \
    ++model.load_method=safe_state_dict \
    ++model.state_dict_path=/mnt/colehawk/bedrock_prod_automodels/stage_2/11B/alexatm/pytorch_model.bin\
