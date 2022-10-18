#This docker file is designed to be build from outside the mstar parent directory 
FROM 747303060528.dkr.ecr.us-east-2.amazonaws.com/mstar-gitlab:master as base

#do requirements first for faster docker builds on code edits
COPY mstar/scripts/encoder_decoder_training/src/requirements.txt  /usr/local/src/mstar/scripts/encoder_decoder_training/src/requirements.txt
#install requirements
RUN pip install -r /usr/local/src/mstar/scripts/encoder_decoder_training/src/requirements.txt

RUN pip install -U pytorch-lightning@git+https://github.com/barry-jin/pytorch-lightning@release/1.6.5#egg=pytorch-lightning

RUN export MAX_NUM_FETCH_EVENT=1
RUN export MAX_NUM_REDUCE_EVENT=1

COPY mstar/ /usr/local/src/mstar/
#Also copy over .git for possible dev
COPY mstar/.git/ usr/local/src/mstar/.git/

WORKDIR /usr/local/src/mstar/scripts/encoder_decoder_training/src/
