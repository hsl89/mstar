#!/bin/bash
NAME='tmp'
MAX_STEPS=5
VAL_INTERVAL=2
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
        optimizer=mtl\
        trainer=mtl\
	model=1_9B\
        data.labeled_data_path=adirawal/p3_packed/oct24_bedrock/ICL/p3-2048-1024-bedrockbytetoken-perturbed1x/\
        optimizer.base_learning_rate=0.0001000\
        optimizer.lr_scheduler_type=constant_with_warmup\
        optimizer.micro_batch_size=${MICRO_BATCH_SIZE}\
        optimizer.labeled_micro_batch_size=${LABELED_MICRO_BATCH_SIZE}\
	optimizer.unlabeled_micro_batch_size=${UNLABELED_MICRO_BATCH_SIZE}\
        optimizer.warmup_steps=${WARMUP_STEPS}\
        optimizer.seed=45\
        optimizer.scheduler_mult_factor=1.00\
        model.positional_embedding=alibi\
        deepspeed_path=config/deepspeed/bf16_zero2d.json\
        trainer.num_nodes=1\
        trainer.precision=bf16\
        trainer.num_sanity_val_steps=0\
        trainer.max_steps=${MAX_STEPS}\
        trainer.log_every_n_steps=1\
        trainer.val_check_interval=${VAL_INTERVAL}\
        trainer.replace_sampler_ddp=False\
        trainer.limit_val_batches=${LIMIT_VAL_BATCHES}\
        lightning.callbacks.checkpoint.every_n_train_steps=2\
        ++average_loss_over_examples=1\
        ++data.autotokenizer_path=/mnt/tokenizer/mstar-t5-sentencepiece-extra_ids_1920-byte_fallback/\
        ++model.state_dict_path=/mnt/adirawal/packaged_models/mstar-t5-1-9B-bedrock/stage_3_t5_bytepatch/pytorch_model.bin\
        model.dropout_rate=0.1\

