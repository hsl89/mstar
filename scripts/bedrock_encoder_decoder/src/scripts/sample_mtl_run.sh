#!/bin/bash
NAME='tmp'
MAX_STEPS=5
CKPT_INTERVAL=4
SAVE_INTERVAL=10000
CKPT='' 
LIMIT_VAL_BATCHES=5 
WARMUP_STEPS=0
MICRO_BATCH_SIZE=4 #per-device batch size
LABELED_MICRO_BATCH_SIZE=3 #per-device batch size
UNLABELED_MICRO_BATCH_SIZE=1 #per-device batch size

python3 pretrain_main.py\
        run_name=${NAME}\
        experiment_name=mtl_validation\
        lightning=mtl\
        data=mtl\
        optimization=mtl\
        optimization/scheduler=constant_with_warmup\
        trainer=mtl\
	model=1_9B\
        data.labeled_data_path=adirawal/p3_packed/oct24_bedrock/ICL/p3-2048-1024-bedrockbytetoken-perturbed1x/\
        optimization.optimizer.lr=0.0001000\
        optimization.micro_batch_size=${MICRO_BATCH_SIZE}\
        optimization.labeled_micro_batch_size=${LABELED_MICRO_BATCH_SIZE}\
	optimization.unlabeled_micro_batch_size=${UNLABELED_MICRO_BATCH_SIZE}\
        optimization.scheduler.num_warmup_steps=${WARMUP_STEPS}\
        optimization.seed=45\
        deepspeed_path=config/deepspeed/bf16_zero2d.json\
        trainer.max_steps=${MAX_STEPS}\
        trainer.log_every_n_steps=1\
        trainer.replace_sampler_ddp=False\
        trainer.limit_val_batches=${LIMIT_VAL_BATCHES}\
        lightning.callbacks.checkpoint.every_n_train_steps=${CKPT_INTERVAL}\
        ++average_loss_over_examples=1\
        ++data.autotokenizer_path=/mnt/tokenizer/mstar-t5-sentencepiece-extra_ids_1920-byte_fallback/\
        ++model.state_dict_path=/mnt/adirawal/packaged_models/mstar-t5-1-9B-bedrock/stage_3_t5_bytepatch/pytorch_model.bin\
        model.dropout_rate=0.1\
