#!/bin/bash
NAME='tmp_bench' #useful to always keep test runs in the same directory to ensure equality
MAX_STEPS=10000
INTERVAL=2000
SAVE_INTERVAL=2000
CKPT='' 

#Usually testing on a g4
python3 pretrain_main.py \
    trainer.max_steps=$MAX_STEPS \
    data.new_datamodule=1 \
    data=t5_reddit \
    data.collator_output_targets=fake_t5 \
    callback.save_every_n_train_steps=${SAVE_INTERVAL} \
    trainer.val_check_interval=${SAVE_INTERVAL} \
    trainer.limit_val_batches=10 \
    run_name=${NAME} \
    trainer.log_every_n_steps=${INTERVAL}  \
    optimizer.warmup_steps=500 \
    model=20B \
    model.positional_embedding=alibi \
    trainer.num_sanity_val_steps=0 \
    deepspeed_path=config/deepspeed/zero2d_shard_16.json \
    optimizer.micro_batch_size=2 \
    optimizer.total_batch_size=1280 \
    trainer.num_nodes=80 \
    
    #model=11B \
    #model.positional_embedding=t5 \
    #trainer.num_sanity_val_steps=0 \
    #deepspeed_path=config/deepspeed/zero2d.json \
    #optimizer.micro_batch_size=2 \
    #optimizer.total_batch_size=16 \
