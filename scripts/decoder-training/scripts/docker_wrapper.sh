#!/bin/bash

TAG="colehawk-dec-01-12"

for region in 'us-west-2' 'us-east-1';
do bash scripts/ec2_docker_build.sh $region $TAG;
done
