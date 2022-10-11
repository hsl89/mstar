#!/bin/bash
#Launches docker file but requires region arg

#args
REGION=$1 #EKS cluster region
TAG=$2 #tag for your dockerfile
DEV=${3-"0"} #Optional. Standard dockerfile by default, dev dockerfile if DEV=1
#end args

echo $DEV

if [ $DEV = '1' ]
then 
    dockerfile_path="mstar/scripts/encoder_decoder_training/dev_Dockerfile"
    TAG="${TAG}_dev"
elif [ $DEV = '0' ]
then
    dockerfile_path="mstar/scripts/encoder_decoder_training/Dockerfile"
fi

echo $dockerfile_path

echo "Creating dockerfile for region ${REGION}"

#Set region for first pull from us-east-2
aws --profile gluonnlp ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 747303060528.dkr.ecr.us-east-2.amazonaws.com

#build docker file from parent directory
cd ../../..
DOCKER_BUILDKIT=1 docker build -t $TAG --build-arg DOCKER_REGION=${REGION} -f $dockerfile_path .
#tag dockerfile
docker tag ${TAG}:latest 747303060528.dkr.ecr.${REGION}.amazonaws.com/mstar-eks:$(whoami)-$TAG

#Set region for push (may be different than pull region)
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin 747303060528.dkr.ecr.${REGION}.amazonaws.com
#push to the cirrus cloud
docker push 747303060528.dkr.ecr.${REGION}.amazonaws.com/mstar-eks:$(whoami)-$TAG

