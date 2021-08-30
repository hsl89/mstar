# Distributed training infra

## Usage guide

This folder contains code provides the M* cluster compatible Docker containers
to run vector/pretraining jobs. You will likely want to modify the Docker
container to contain your updated code, instead of relying on the code from the
master branch.

### Configure mstar account on your local machine

Please install the isengard tool, if you haven't done so already:

```
python3 -m pip install --upgrade --user git+ssh://git.amazon.com/pkg/BenderLibIsengard
```

Please add the following section to your `~/.aws/config` file:

```
[profile mstar]
region=us-east-1
account=216212465934
role=PowerUser
credential_process = /path/to/installed/isengard get 216212465934 PowerUser --json
```

### Setting up AWS Batch to use your own Docker

To run experiments on the cluster, please create your own AWS Batch
JobDefinition by running the following commands

```
cd ~/src/vector-science/pretraining/mstar-cluster
aws --profile mstar --region us-east-1 cloudformation create-stack \
    --stack-name mstar-mnp-jobdefinition-$(whoami) \
    --template-body file://$(pwd)/cfn_jobdefinition.yaml \
    --parameter ParameterKey=User,ParameterValue=$(whoami) ParameterKey=ClusterStackName,ParameterValue=mstar-mnp
```

### Building your own Docker and publishing it for use on the cluster

**WHEN RUNNING THE BUILD ON AN EC2 MACHINE, BE SURE TO REPLACE $(whoami) WITH YOUR AMAZON LOGIN**

Retrieve an authentication token and authenticate your Docker client to your
registry. Use the AWS CLI:

`aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 216212465934.dkr.ecr.us-east-1.amazonaws.com`

Build your Docker image using the following command from the root of the vector
repository. For information on building a Docker file from scratch see the
instructions here You can skip this step if your image is already built:

`DOCKER_BUILDKIT=1 docker build -t mstar-mnp -f mstar-cluster/Dockerfile .`

After the build completes, tag your image so you can push the image to this repository:

`docker tag mstar-mnp:latest 216212465934.dkr.ecr.us-east-1.amazonaws.com/mstar-mnp:$(whoami)`

Run the following command to push this image to your newly created AWS repository:

`docker push 216212465934.dkr.ecr.us-east-1.amazonaws.com/mstar-mnp:$(whoami)`

### Submitting jobs

You can submit jobs using the awscli as in the following example:

```
aws --profile mstar --region us-east-1 batch submit-job \
    --job-queue mstar-mnp-p4 --job-name $(whoami) --job-definition mstar-mnp-$(whoami)-p4 \
    --node-overrides '{"numNodes": 8, "nodePropertyOverrides": [{"targetNodes": "0:", "containerOverrides": {"environment": [{"name":"FI_PROVIDER", "value":"efa"}, {"name":"FI_EFA_USE_DEVICE_RDMA", "value":"1"}], "command": ["sh", "-c", "python3 scripts/pretraining/pretrain_main.py --model_type vectorbart --training_dataset /mnt/training.arrow --validation_dataset /mnt/validation.arrow --config_path scripts/pretraining/configs/mbart_base.json  --tokenizer_path /mnt/zijwan/model_testing/spm_0721_converted  --max_seq_length 512 --warmup_steps 5000 --val_check_interval 10000 --batch_size 32 --multilang_sampling_alpha 0.3  --gpus 8 --accelerator deepspeed --precision 16 --max_steps 100000 --default_root_dir $VECTOR_DEFAULT_ROOT_DIR"]}}]}'
```

Note that the command doesn't use Deepspeed, as we currently face a Deepspeed crash in the multi-node setting (see Appendix).

**Be sure to remember the jobId returned.**

### Logs
To follow live logs of your experiments, using awscli v2:

```
aws --profile mstar --region us-east-1 logs tail /mstar-mnp-$(whoami)-p4/log --follow --since 12h
```

For Tensorboard, just run the following command, replacing JOBID with the jobId obtained when submitting the job

```
AWS_PROFILE=mstar tensorboard --logdir s3://mstar-mnp-dev/batch/mstar-mnp-p4/JOBID/
```

## FAQ
### My run crashes with `cudaErrorIllegalAddress: an illegal memory access was encountered`

We have experienced cudaErrorIllegalAddress due to obscure hardware issues.
Please see https://t.corp.amazon.com/V418224885 for details and report the issue
to the rest of the team. The instance you observed the error may need to be
isolated.


## Guide for maintainers

### AWS Infra

`cloudformation_template.yaml` provides a template to prepare a AWS Batch
infrastructure in an AWS account. A new account can be setup via `aws
cloudformation create-stack --stack-name mstar-mnp --template-body
file://$(pwd)/cloudformation_template.yaml`.

Note that the `stack-name` must be unique.

### ECR

The template currently sets up an ECR repository to which containers for use in AWS Batch can be pushed.

Retrieve an authentication token and authenticate your Docker client to your
registry. Build your Docker image using the following command from the root of
the mstar repository. After the build completes, tag your image so you can push
the image to this repository.

The commands depend on the account and region of the cluster.

M* us-east-1
```
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 216212465934.dkr.ecr.us-east-1.amazonaws.com
DOCKER_BUILDKIT=1 docker build -t mstar-mnp -f tools/distributed-training/Dockerfile .
docker tag mstar-mnp:latest 216212465934.dkr.ecr.us-east-1.amazonaws.com/mstar-mnp:latest
docker push 216212465934.dkr.ecr.us-east-1.amazonaws.com/mstar-mnp:latest
```

M5 us-east-2
```
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 350694149704.dkr.ecr.us-east-2.amazonaws.com
DOCKER_BUILDKIT=1 docker build -t mstar-mnp -f tools/distributed-training/Dockerfile .
docker tag mstar-mnp:latest 350694149704.dkr.ecr.us-east-2.amazonaws.com/mstar-m5-us-east-2:latest
docker push 350694149704.dkr.ecr.us-east-2.amazonaws.com/mstar-m5-us-east-2:latest
```
