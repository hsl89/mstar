#!/bin/bash
#Launches docker build
#Moves to the parent directory of mstar for the build command

DOCKERFILE_PATH="mstar/scripts/bedrock_encoder_decoder/Dockerfile"

TAG=$1

echo $TAG
echo $DOCKERFILE_PATH

#Set region for first pull
#assumes build from us-east-2 master container
aws ecr get-login-password --region us-east-2 | sudo docker login --username AWS --password-stdin 747303060528.dkr.ecr.us-east-2.amazonaws.com

#build docker file from parent directory of mstar
cd ../../..
echo $TAG
DOCKER_BUILDKIT=1 sudo docker build --no-cache -t $TAG -f $DOCKERFILE_PATH . 

#to avoid credential overlap for different ecr repos
sudo docker logout
