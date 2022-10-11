#This docker file is designed to be build from outside the mstar parent directory 
FROM 747303060528.dkr.ecr.us-east-2.amazonaws.com/mstar-gitlab:master as base

#do requirements first for faster docker builds on code edits
COPY mstar/scripts/encoder_decoder_training/src/requirements.txt  /usr/local/src/mstar/scripts/encoder_decoder_training/src/requirements.txt
#install requirements
RUN pip install -r /usr/local/src/mstar/scripts/encoder_decoder_training/src/requirements.txt

COPY mstar/ /usr/local/src/mstar/

WORKDIR /usr/local/src/mstar/scripts/encoder_decoder_training/src/
