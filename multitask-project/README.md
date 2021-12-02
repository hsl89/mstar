## Setup

Run the setup per the main readme of the mstar-milestone-1 repo. Data can be obtained as:
```
mkdir data
aws s3 cp s3://padmakv-data/copy_data.sh .
sh copy_data.sh
aws s3 cp s3://padmakv-data/metadata-with-kind/ ./data/metadata-with-kind --recursive
```

## Pre-finetuning
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node 4 run_prefinetuning.py --data-args ./data/metadata-with-kind/large_args_all --model-type xlm-roberta-base --save-model test-sl-qa-glue
```
The JSON files for arguments to run pre-finetuning on the various subsets are in the folder ./data/metadata-with-kind. These can also be modified easily to try out other subsets of tasks. 

## Target Task Finetuning

### SL Tasks
This trains a single task model on the concerned SL task. You can also change the model type with --model-type
```
python3 finetune_model_sl.py --target-task-args ./data/metadata-with-kind/$task.json --num-epochs 40 > results/$task
```
And to run target task finetuning on a pre-finetuned model use this run command (ensure that you match the right arguments file)
```
python3 finetune_model_sl.py --target-task-args ./data/metadata-with-kind/$task.json --num-epochs 40 --use-encoder-weights ./model_dir --prefinetuning-args arguments_file  > results/$task
```

### Superglue Tasks 
Same format as SL tasks
```
python3 finetune_model_superglue.py --target-task-args ./data/metadata-with-kind/$task.json --num-epochs 40 > results/$seed/$task

python3 finetune_model_superglue.py --target-task-args ./data/metadata-with-kind/$task.json --num-epochs 40 --use-encoder-weights ./model_dir --prefinetuning-args arguments_file  > results/$task
```

### QA Tasks
Same format as SL and Superglue tasks
```
python3 finetune_model_qa.py --target-task-args ./data/metadata-with-kind/$task.json --num-epochs 40 > results/$seed/$task

python3 finetune_model_superglue.py --target-task-args ./data/metadata-with-kind/$task.json --num-epochs 40 --use-encoder-weights ./model_dir --prefinetuning-args arguments_file  > results/$seed/$task
```

## Benchmarking all unseen tasks
Retrieve the pre-finetuned models
```
aws s3 cp s3://padmakv-data/Pre-Finetuned-Models/ ./saved-models/ --recursive
```
And then benchmark each model with the following script, retrieve the model arguments from ./data/metadata-with-kind or from loading the actual pytorch models and retrieving model.mtl\_args
```
sh unseen_task_benchmark.sh
```

## Notes on UF
uf\_reader is lifted from the Comprehend code, and the relevant dependencies are moved into the dependencies folder. The script runs with only imports from this repository now.
