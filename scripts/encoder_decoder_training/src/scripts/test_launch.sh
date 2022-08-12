#!/bin/bash
NAME='tmp_1' #useful to always keep test runs in the same directory to ensure equality
MAX_STEPS=100
INTERVAL=10
SAVE_INTERVAL=10 

python3 pretrain_main.py trainer.max_steps=$MAX_STEPS model.ckpt_path=$ckpt callback.save_every_n_train_steps=${SAVE_INTERVAL} trainer.val_check_interval=${SAVE_INTERVAL} trainer.limit_val_batches=${INTERVAL} run_name=${NAME} trainer.log_every_n_steps=${INTERVAL} 

