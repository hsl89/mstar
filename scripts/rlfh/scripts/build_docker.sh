FILE_NAME=$1
AWS_DEFAULT_REGION=$2
DOCKER_NAME=$3
INSTANCE=$4
echo "Example usage: \n cd ~/mstar \n sh ./scripts/rlfh/scripts/build_docker.sh Dockerfile_p3 us-east-2 kaixianl-rlfh p3"
cp ./scripts/rlfh/requirements.txt ./requirements.txt
DOCKER_BUILDKIT=1 docker build -t mstar-$INSTANCE --target base -f ./scripts/rlfh/$FILE_NAME .
docker tag mstar-$INSTANCE:latest 747303060528.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/mstar-eks:$DOCKER_NAME-$INSTANCE
docker push 747303060528.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com/mstar-eks:$DOCKER_NAME-$INSTANCE
