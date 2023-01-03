#!/bin/bash
#need to pass this in and have it be unique each time
ID=$1

MAX_STEPS=1000
CKPT_INTERVAL=100
CKPT='' 
LIMIT_VAL_BATCHES=10 
WARMUP_STEPS=100
MICRO_BATCH_SIZE=2 #per-device batch size
NUM_WORKERS=32
PREFETCH_FACTOR=4
CKPT_INTERVAL=1000
NAME="tmp_elastic_2B"

#faster to do setup on 2 gpus
GPUS=2
export CUDA_VISIBLE_DEVICES=0,1
export REAL_WORLD_SIZE=${GPUS}


python3 -m torch.distributed.run --nnodes=1 --nproc_per_node=${GPUS} --rdzv_id=${ID} --rdzv_backend=etcd --rdzv_endpoint=10.100.66.79:2379 pretrain_main.py \
    lightning/plugins/environment=torch_elastic \
    optimization.micro_batch_size=2\
    data.clm_max_doc=1\
    model.ckpt_path='auto' \
    model=770M \
    data=tmp\
    deepspeed_path=config/deepspeed/bf16_zero2d_shard2.json \
    data.num_workers=${NUM_WORKERS} \
    data.prefetch_factor=${PREFETCH_FACTOR} \
    trainer.log_every_n_steps=1\
    lightning.callbacks.progress_bar.refresh_rate=1\
    max_steps=$MAX_STEPS \
    lightning.callbacks.checkpoint.every_n_train_steps=$CKPT_INTERVAL \
    trainer.limit_val_batches=${LIMIT_VAL_BATCHES} \
    run_name=${NAME} \
    optimization.micro_batch_size=${MICRO_BATCH_SIZE} \
    optimization.scheduler.num_warmup_steps=${WARMUP_STEPS} \
    optimization.scheduler.scale_factor=${WARMUP_STEPS} \
    trainer.reload_dataloaders_every_n_epochs=0 \
