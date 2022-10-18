#!/bin/bash

git pull 

DIRNAME="/tmp_mstar"
mkdir $DIRNAME
cd $DIRNAME
git clone https://gitlab.aws.dev/mstar/mstar.git

cd mstar


#add nvcc to path
export PATH=/usr/local/cuda/bin:$PATH

pip install -U -e . --no-build-isolation
