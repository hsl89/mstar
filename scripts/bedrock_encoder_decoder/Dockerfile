#This docker file is designed to be build from outside the mstar parent directory 
FROM 747303060528.dkr.ecr.us-east-2.amazonaws.com/mstar-gitlab:master as base

#copy over full local repo
COPY mstar/ /usr/local/src/mstar/

#Also copy over .git for possible dev
COPY mstar/.git/ /usr/local/src/mstar/.git/

#install extra requirements, specifically apex
#apex required for mstar.optimizers.FusedAdam and
#used for RMSNorm
#needs to be installed before installing extras
COPY mstar/scripts/bedrock_encoder_decoder/src/requirements.txt  /usr/local/src/mstar/scripts/bedrock_encoder_decoder/src/requirements.txt
RUN pip install -r /usr/local/src/mstar/scripts/bedrock_encoder_decoder/src/requirements.txt

# install requirements via M* setup
# need to add path for NVCC,
# make sure we compile megatron kernels wiwth FORCE_CUDA=1
# TODO(colehawk) avoid re-installation of the CUDA kernels but keep apex
RUN cd /usr/local/src/mstar/ && \
	export PATH=/usr/local/cuda/bin/:$PATH && \
	export FORCE_CUDA=1  && \
	export TORCH_CUDA_ARCH_LIST="8.6 8.0" && \
	pip install -U -e .[bedrock_encoder_decoder] --no-build-isolation

#TODO(colehawk) move main branch to 1.8.6
RUN pip install -U pytorch_lightning==1.8.6

#set working directory for job runs
WORKDIR /usr/local/src/mstar/scripts/bedrock_encoder_decoder/src/
