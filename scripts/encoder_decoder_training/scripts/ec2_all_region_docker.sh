#!/bin/bash
#Launches docker file but requires region arg and name

TAG='easel'
REGIONS=('us-east-1' 'us-east-2' 'us-west-2' 'ap-northeast-2')
DEVS=('0') #set to ('0' '1') to add dev dockerfile
NAME='YOURNAME'

for DEV in ${DEVS[@]};
do bash scripts/ec2_build_docker.sh $TAG $DEV;
done 

for REGION in ${REGIONS[@]};
do for DEV in ${DEVS[@]};
do bash scripts/ec2_push_docker.sh $REGION $TAG $DEV $NAME;
done 
done
