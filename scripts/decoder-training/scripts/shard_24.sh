#!/bin/bash

python3 pretrain_main.py \
    trainer.default_root_dir=/mnt_out/colehawk/shard_24 \
    model.fused_scaled_masked_softmax=1\
    model.gradient_checkpointing=1\
    deepspeed_config=config/deepspeed/stage2d-bf16-shard24.json \
    model.arch_path=config/model/architecture/gpt2_52B.json\
    model.positional_embedding=absolute \
    optimizer.batch_size=1 \
    data.max_seq_length=2048 \
    data.max_valid_batch=2 \
    trainer.limit_val_batches=10 \
    model.ckpt_path=auto\
    callback.save_every_n_steps=20 \
    trainer.val_check_interval=19 \
    data.num_workers=1 \
    data.training_dataset=/mnt/pretraining-data/package-01-06-23-v1/val.arrow \
    data.validation_dataset=/mnt/pretraining-data/package-01-06-23-v1/val.arrow \
    #/mnt_out/colehawk/shard_24/2524/ab7f56560db54714977f8ac16a3745cd/checkpoints/epoch\\=0-step\\=60.ckpt/\
    #data.data_state_path=/mnt_out/colehawk/shard_24/2524/ab7f56560db54714977f8ac16a3745cd/checkpoints/epoch\\=0-step\\=60.ckpt/checkpoint/zero_pp_rank_0_mp_rank_00_model_states.pt \
