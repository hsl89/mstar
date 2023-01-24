# Emebedding model distributed evaluation pipeline
The pipeline uses [MTEB](https://github.com/hsl89/mteb) for evaluating embedding models with distributed inference.
The image of the current stable pipeline
```
747303060528.dkr.ecr.us-east-1.amazonaws.com/mstar-gitlab:encoder-eval-7.3
```

## Build the image for the pipeline
```
export REGION=us-east-1 # where to host the image
export TAG=7.2 # tag of the image
bash build.sh $REGION $TAG
```

## Usage
We currently support sts, retrieval and clustering task. Please refer to this table to check the name of the task. 
Command line arguments is managed by [hydra](https://hydra.cc/docs/intro/). To evaluate one or more task (of the same task type)
```
export PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=16 
python3 main.py model=<model_type> model.model_params.model_name=<model_name> \ # refer to conf/model
    model.tokenizer_params.tokenizer_name=<tokenizer name> \ 
    model.data_params.batch_size=<batch size> \ # to override default batch size for the model defined under conf/model
    task=<task type> \ # refer to conf/task
    task.task_name=[<task_name1>, <task_name2>,...] \
    split=[<data split1>,<data_split2>] #  
```
For example, if you want to evaluate simcse model on 2 retrieval tasks with test split: FiQA2018 and SCIDOCS using a batch size of 128
```
export PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=16 
python3 main.py model=sup_simcse model.data_params.batch_size=128 task=retrieval task.task_name=[FiQA2018,SCIDOCS] split=[test] 
```

If you want to evaluate an mstar model with `model_name=mstar-bert-5B-bedrock-expmodel-20221110-weightaverage-mtlv2K1M-msmarcosupcl`
on the above tasks
```
export PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=16 
python3 main.py model=mstar \
    model.model_params.model_name=model_name=mstar-bert-5B-bedrock-expmodel-20221110-weightaverage-mtlv2K1M-msmarcosupcl \
    model.data_params.batch_size=4 task=retrieval task.task_name=[FiQA2018,SCIDOCS] split=[test] 
```

You get it, just look at `conf/model` to see how each model type is defined. From there, you can figure out how to overwrite each argument. 

The EKS job submission script for the above job would be
```
name: retrieval
node_type: "p4d.24xlarge"
node_num: 1
image: "747303060528.dkr.ecr.us-east-1.amazonaws.com/mstar-gitlab:encoder-eval-7.0"
command: ["bash", "-c"]
args: [
    "PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=16 python3 mstar/scripts/encoder-eval/main.py model=sup_simcse task=retrieval task.task_name=[FiQA2018,SCIDOCS] split=[test]"
]                                                                                            
```

### Evaluating sgpt model
You need to specify whether you want to use [asymmetric search or not](https://github.com/Muennighoff/sgpt#asymmetric-semantic-search-be) while evaluating sgpt model. For asymmetric search, queries are wrapped in `[]` and docs are wrapped in `{}`; for symmetric search no preprocessing is applied to queries or documents.

Example for evaluating SGPT model with asymmetric search
```sh
PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=16 python3 mstar/scripts/encoder-eval/main.py model=spgt model.model_params.model_name=Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit model.model_params.asymmetric_search=True model.tokenizer_params.tokenizer_name=Muennighoff/SGPT-125M-weightedmean-msmarco-specb-bitfit task=retrieval task.task_name=[FiQA2018,SCIDOCS] split=[test] 
```

Example for evaluate SGPT model with symmetric search
```sh
PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=16 python3 mstar/scripts/encoder-eval/main.py model=spgt model.model_params.model_name=Muennighoff/SGPT-125M-weightedmean-nli-bitfit model.model_params.asymmetric_search=False model.tokenizer_params.tokenizer_name=Muennighoff/SGPT-125M-weightedmean-nli-bitfit task=retrieval task.task_name=[FiQA2018,SCIDOCS] split=[test] 
```

## Test
we decouple the pipeline's capability to do inference and everything else (data loading, model loading, integration with MLflow etc)
the reason is that inference is the most costly step, once its correctness is verifed,
we can run integration test by using fake inference, i.e. generating random vectors in the encoding stage
in `test.py`, we only test our pipeline's capability to do distributed inference
see `integ_test.sh` for integration test

To test model inference
```
python3 test.py
```
To run integration test
```
bash integ_test.sh
```






