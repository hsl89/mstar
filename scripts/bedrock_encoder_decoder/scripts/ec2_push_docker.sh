#!/bin/bash
#Push docker image to a specific region

REGION=$1
TAG=$2

echo "Pushing dockerfile to region ${REGION}"

sudo docker tag ${TAG}:latest 747303060528.dkr.ecr.${REGION}.amazonaws.com/mstar-eks:$TAG
#Set region for push (may be different)
aws ecr get-login-password --region $REGION | sudo docker login --username AWS --password-stdin 747303060528.dkr.ecr.${REGION}.amazonaws.com
sudo docker push 747303060528.dkr.ecr.${REGION}.amazonaws.com/mstar-eks:$TAG

#avoid credential overlap for different regions
sudo docker logout
