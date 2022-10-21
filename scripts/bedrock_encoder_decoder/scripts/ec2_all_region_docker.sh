#!/bin/bash
#Launches docker file but requires region arg

TAG='bedrock'
REGIONS=('us-east-1' 'us-east-2' 'us-west-2' 'ap-northeast-2')
DEVS=('0') #add '1' to array for dev dockerfile
NAME='colehawk'

#force clear everything out, rebuild the cache
#sudo docker system prune -a -f

for DEV in ${DEVS[@]};
do bash scripts/ec2_build_docker.sh $TAG $DEV;
done 

for REGION in ${REGIONS[@]};
do for DEV in ${DEVS[@]};
do bash scripts/ec2_push_docker.sh $REGION $TAG $DEV $NAME;
done 
done
