## Base docker image built on Dockerfile.mstar.base
FROM 747303060528.dkr.ecr.us-east-2.amazonaws.com/mstar-gitlab:base as base
LABEL maintainer="MStar Team"

# Copy the source code and install 
RUN mkdir -p /usr/local/src/mstar
COPY . /usr/local/src/mstar/
RUN cd /usr/local/src/mstar && \
    TORCH_CUDA_ARCH_LIST="8.0 7.5 7.0" FORCE_CUDA=1 python3 -m pip install /usr/local/src/mstar --no-build-isolation

WORKDIR /usr/local/src/mstar
COPY tools/distributed-training/entrypoint.sh /usr/local/entrypoint.sh
ENTRYPOINT [ "/usr/local/entrypoint.sh" ]
