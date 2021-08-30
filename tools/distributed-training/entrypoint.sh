#!/usr/bin/env bash


# Set up environment variables
if [[ -n $AWS_BATCH_JOB_ID ]]; then
    # Map BATCH_JOB_ID:ARRAY_JOB_ID format to BATCH_JOB_ID/ARRAY_JOB_ID
    MY_JOB_ID=${AWS_BATCH_JOB_ID//:/\/}
    # Map BATCH_JOB_ID#AWS_BATCH_JOB_NODE_INDEX format to BATCH_JOB_ID/AWS_BATCH_JOB_NODE_INDEX
    MY_JOB_ID=${MY_JOB_ID//#/\/}
    # Expose S3 path in environment
    export MSTAR_DEFAULT_ROOT_DIR=s3://mstar-us-west-2-mnp-dev/batch/$AWS_BATCH_JQ_NAME/${MY_JOB_ID}/

    # Configure NCCL and Gloo for AWS Batch
    export NCCL_SOCKET_IFNAME=eth0
    export GLOO_SOCKET_IFNAME=eth0
else
    export MSTAR_DEFAULT_ROOT_DIR=./
fi;

# Print system config
echo "% ip addr"
ip addr
echo
echo

echo "% fi_info -p efa -t FI_EP_RDM"
fi_info -p efa -t FI_EP_RDM
echo
echo


echo "% printenv"
printenv
echo
echo

exec "$@"
