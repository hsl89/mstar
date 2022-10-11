#!/bin/bash
#Launches docker file but requires region arg

REGION=$1
TAG=$2
DEV=${3-"0"}
NAME=$4

#assumes the build is also done with the dev tag
if [ $DEV = '1' ]
then 
    TAG="${TAG}_dev"
fi


echo "Pushing dockerfile to region ${REGION}"

sudo docker tag ${TAG}:latest 747303060528.dkr.ecr.${REGION}.amazonaws.com/mstar-eks:${NAME}-$TAG
#Set region for push (may be different)
aws ecr get-login-password --region $REGION | sudo docker login --username AWS --password-stdin 747303060528.dkr.ecr.${REGION}.amazonaws.com
sudo docker push 747303060528.dkr.ecr.${REGION}.amazonaws.com/mstar-eks:${NAME}-$TAG


#avoid credential overlap for different regions
sudo docker logout
