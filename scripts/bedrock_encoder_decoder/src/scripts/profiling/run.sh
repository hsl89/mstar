#!/bin/bash
NUM_GPUS=8
DATA_DIR='/mnt/colehawk/pile_no_youtube' #where are train/test/val datasets

INDEX_DIR='/mnt/colehawk/pile_no_youtube/indices/256_gpu/' #where the index files are
BATCH_SIZE=4
MAX_STEPS=3 #this is enough for the TFLOPS count to stabilize on a single node

#test custom model matches
#python test_custom_model_eq_outputs.py

#then run profiling
NAME=flops_test

#see explanation of nsys options here
#https://gist.github.com/mcarilli/376821aa1a7182dfcf59928a7cde3223

nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s none -o nsight/nsight_report -f true --capture-range=cudaProfilerApi --cudabacktrace true -x true python3 profile_pretrain.py \
        --run_name $NAME \
        --deepspeed_config deepspeed/stage2.json \
	--num_workers 96 \
        --lr_scheduler_type linear \
        --base_learning_rate 0.0001 \
        --model_type t5 \
        --precision 16 \
        --optimizer fusedadam \
	--training_dataset ${DATA_DIR}/train_packed_chunksize_2600.arrow \
	--validation_dataset ${DATA_DIR}/val_packed_chunksize_2600.arrow \
	--train_data_index_path ${INDEX_DIR}/train/ \
	--valid_data_index_path ${INDEX_DIR}/val/ \
	--config_path config/model/t5_wide/1_3B.json \
	--gpus $NUM_GPUS \
	--batch_size $BATCH_SIZE \
	--max_seq_length 2048 \
	--activation_checkpointing 1 \
        --gradient_clip_val 1.0 \
	--strategy deepspeed \
	--default_root_dir /mnt_out/colehawk/easel/flops_test\
        --weight_decay 0.1 \
        --replace_sampler_ddp False \
        --save_top_k 0 \
	--save_every_n_steps 5000 \
        --limit_val_batches 2 \
        --max_steps $MAX_STEPS \
        --base_max_steps $MAX_STEPS \
        | tee logs/$NAME.txt
