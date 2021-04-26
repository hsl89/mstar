FROM ubuntu:20.04 as base

LABEL maintainer="mstar team"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    SHELL=/bin/bash

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y curl wget python3 python3-pip build-essential ninja-build git unzip zstd && \
    rm -rf /var/lib/apt/lists/*

RUN cd /usr/local/src && git clone https://github.com/dmlc/gluon-nlp && \
    python3 -m pip install torch==1.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    python3 -m pip install --pre mxnet && \
    python3 -m pip install -e /usr/local/src/gluon-nlp'[extras]' && \
    python3 -m pip install transformers awscli

ADD batch/job.sh /usr/local/bin/job.sh
ADD prepare_quickthought.py /usr/local/bin/prepare_quickthought.py