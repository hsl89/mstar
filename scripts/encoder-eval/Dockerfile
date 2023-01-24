ARG REGION 
ARG BASE=747303060528.dkr.ecr.$REGION.amazonaws.com/mstar-gitlab:encoder-eval-base as base
FROM $BASE
COPY . /usr/local/src/mstar/

WORKDIR /usr/local/src/

ARG CACHEBUST=1
RUN git clone https://github.com/hsl89/mteb.git && \
  cd mteb && python3 setup.py develop 

RUN python3 -m pip install flatten-dict

