#This docker file is designed to be build from outside the mstar parent directory 
FROM 747303060528.dkr.ecr.us-east-2.amazonaws.com/mstar-gitlab:master as base

# copy mstar mainline
COPY mstar/ /usr/local/src/mstar/

# copy latest source code for decoder training
COPY mstar/scripts/decoder-training/ /usr/local/src/mstar/scripts/decoder-training/
#install requirements
RUN pip install -r /usr/local/src/mstar/scripts/decoder-training/requirements.txt
# overwrite mstar lightning version, needed for zero-2d
#RUN python3 -m pip install pytorch-lightning@git+https://github.com/barry-jin/pytorch-lightning@release/1.6.5#egg=pytorch-lightning --no-build-isolation --force-reinstall --no-deps
RUN pip install -U pytorch_lightning==1.8.6

WORKDIR /usr/local/src/mstar/scripts/decoder-training/

