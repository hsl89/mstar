#!/bin/bash
#Launches docker file but requires region arg

#args
REGION=$1 #EKS cluster region
TAG=$2 #tag for your dockerfile

dockerfile_path="mstar/scripts/decoder-training/Dockerfile"

echo $dockerfile_path

echo "Creating dockerfile for region ${REGION}"

#Set region for first pull from us-east-2
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 747303060528.dkr.ecr.us-east-2.amazonaws.com

#build docker file from parent directory
cd ../../../
DOCKER_BUILDKIT=1 docker build -t ${TAG}  -f ${dockerfile_path} .
#tag dockerfile
docker tag ${TAG}:latest 747303060528.dkr.ecr.${REGION}.amazonaws.com/mstar-eks:${TAG}

#Set region for push (may be different than pull region)
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin 747303060528.dkr.ecr.${REGION}.amazonaws.com
#push to the cirrus cloud
docker push 747303060528.dkr.ecr.${REGION}.amazonaws.com/mstar-eks:${TAG}
cd -

echo pushed the following docker image:
echo 747303060528.dkr.ecr.${REGION}.amazonaws.com/mstar-eks:${TAG}

