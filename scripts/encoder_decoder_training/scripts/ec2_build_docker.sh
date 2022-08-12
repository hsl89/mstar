#!/bin/bash
#Launches docker file but requires region arg

TAG=$1
DEV=${2-"0"} #can be used to specify a diferent development dockerfile

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

#Set region for first pull
aws ecr get-login-password --region us-east-2 | sudo docker login --username AWS --password-stdin 747303060528.dkr.ecr.us-east-2.amazonaws.com


#build docker file from parent directory
cd ../../..
DOCKER_BUILDKIT=1 sudo docker build -t $TAG -f $dockerfile_path .

#to avoid credential overlap for different ecr repos
sudo docker logout
