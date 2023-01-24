# build docker image for IR eval
#!/bin/bash

REGION=${1:-us-east-1}
TAG=${2:-""} # tag of the image
SKIP_BASE=${3:-true} # skip building image 

cd ../../..

export BASE_IMAGE_NAME=747303060528.dkr.ecr.$REGION.amazonaws.com/mstar-gitlab:encoder-eval-base
export SCRIPT_IMAGE_NAME=747303060528.dkr.ecr.$REGION.amazonaws.com/mstar-gitlab:encoder-eval-$TAG

aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin 747303060528.dkr.ecr.$REGION.amazonaws.com


if [[ $SKIP_BASE = true ]]; then
    echo "skip building base image"
else
  docker build -t $BASE_IMAGE_NAME mstar --build-arg REGION=$REGION -f mstar/scripts/encoder-eval/Dockerfile.base
  docker push $BASE_IMAGE_NAME
fi

echo "building script image: $SCRIPT_IMAGE_NAME"
docker build -t $SCRIPT_IMAGE_NAME mstar --build-arg REGION=$REGION --build-arg BASE=$BASE_IMAGE_NAME --build-arg CACHEBUST=$(date +%s) -f mstar/scripts/encoder-eval/Dockerfile 

docker push $SCRIPT_IMAGE_NAME



