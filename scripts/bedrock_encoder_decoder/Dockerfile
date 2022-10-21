#This docker file is designed to be build from outside the mstar parent directory 
FROM 747303060528.dkr.ecr.us-east-2.amazonaws.com/mstar-gitlab:master as base

#copy requirements first for faster docker rebuild
COPY mstar/scripts/bedrock_encoder_decoder/src/requirements.txt  /usr/local/src/mstar/scripts/bedrock_encoder_decoder/src/requirements.txt

#install requirements
RUN pip install -r /usr/local/src/mstar/scripts/bedrock_encoder_decoder/src/requirements.txt

RUN pip install -U pytorch-lightning@git+https://github.com/barry-jin/pytorch-lightning@release/1.6.5#egg=pytorch-lightning

#copy over code in a later layer for faster rebuild
COPY mstar/ /usr/local/src/mstar/

#Also copy over .git for possible dev
COPY mstar/.git/ /usr/local/src/mstar/.git/

WORKDIR /usr/local/src/mstar/scripts/bedrock_encoder_decoder/src/
