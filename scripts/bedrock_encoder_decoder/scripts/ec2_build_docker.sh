#!/bin/bash
#Launches docker file but requires region arg

TAG=$1
DEV=${2-"0"}

echo $DEV

if [ $DEV = '1' ]
then 
    dockerfile_path="mstar/scripts/bedrock_encoder_decoder/dev_Dockerfile"
    TAG="${TAG}_dev"
    BASE_CONTAINER=''
elif [ $DEV = '0' ]
then
    dockerfile_path="mstar/scripts/bedrock_encoder_decoder/Dockerfile"
    BASE_CONTAINER="colehawk-${TAG}"
fi
echo $TAG

echo $dockerfile_path

#Set region for first pull
aws ecr get-login-password --region us-east-2 | sudo docker login --username AWS --password-stdin 747303060528.dkr.ecr.us-east-2.amazonaws.com

########################################################################
# @colehawk-specific install
########################################################################
if [ $DEV = '1' ]
then 
	git clone https://github.com/colehawkins/.dotfiles.git
	cd .dotfiles
	git checkout docker
	cd ..
fi
########################################################################
# @colehawk-specific install
########################################################################

#build docker file from parent directory
cd ../../..
echo $TAG
DOCKER_BUILDKIT=1 sudo docker build -t $TAG -f $dockerfile_path .

########################################################################
# @colehawk-specific install
########################################################################
if [ $DEV = '1' ]
then 
    #cleanup
    rm -rf  mstar/scripts/bedrock_encoder_decoder/.dotfiles
fi
########################################################################
# @colehawk-specific install
########################################################################

#to avoid credential overlap for different ecr repos
sudo docker logout
