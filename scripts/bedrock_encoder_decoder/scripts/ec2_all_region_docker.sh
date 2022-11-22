#!/bin/bash
#Builds + pushes docker images to several regions
#The actual docker build takes place from the parent directory of mstar
#Usage: bash scripts/ec2_all_region_docker.sh colehawk test

USER=$1
DESC=${2-"bedrock"}
DATE=$(date +"%m-%d")
#append date for reproducibility
TAG="${USER}-${DESC}-${DATE}"

echo "Image tag " $TAG

REGIONS=('us-east-1' 'us-east-2' 'us-west-2' 'ap-northeast-2')

#force clear everything out, rebuild the cache
#otherwise changes from master aren't captured
sudo docker system prune -a -f

bash scripts/ec2_build_docker.sh $TAG;

#push to all regions
for REGION in ${REGIONS[@]};
do bash scripts/ec2_push_docker.sh $REGION $NAME $TAG;
done
