# Distributed tools for scripts/bert

## Setup for reproducing

### 0. Setting up AWS Batch cluster via cloudformation_template.yaml`

Set up AWS Batch environment etc.

```
aws --profile mstar cloudformation create-stack --stack-name lausen-mstar --template-body file:///home/ANT.AMAZON.COM/lausen/src/mstar/scripts/bert/batch/cloudformation_template.yaml --capabilities CAPABILITY_NAMED_IAM
```

### 1. Updating the Docker

Currently AWS Batch does not support specifying the Docker image at runtime, but
will always use the image tag specified in the `cloudformation_template.yaml`.
Thus, build the correct image here and push it to the standardized tag.

```
cd ~/src/mstar/scripts/bert

aws --profile mstar ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 216212465934.dkr.ecr.us-east-1.amazonaws.com

docker build -t lausen-mstar . -f batch/Dockerfile

docker tag lausen-mstar:latest 216212465934.dkr.ecr.us-east-1.amazonaws.com/lausen-mstar:latest

docker push 216212465934.dkr.ecr.us-east-1.amazonaws.com/lausen-mstar:latest
```
