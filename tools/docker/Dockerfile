## Base docker image built on Dockerfile.mstar.base
FROM 747303060528.dkr.ecr.us-east-2.amazonaws.com/mstar-gitlab:base1.6 as base
LABEL maintainer="MStar Team"

# Install zero-2d compatible pytorch-lightning
RUN python3 -m pip install pytorch-lightning@git+https://github.com/barry-jin/pytorch-lightning@release/1.6.5#egg=pytorch-lightning --no-build-isolation --force-reinstall --no-deps

# Copy the source code and install 
RUN mkdir -p /usr/local/src/mstar
COPY . /usr/local/src/mstar/
RUN cd /usr/local/src/mstar && \
    TORCH_CUDA_ARCH_LIST="8.0 7.5 7.0" FORCE_CUDA=1 python3 -m pip install /usr/local/src/mstar --no-build-isolation

WORKDIR /usr/local/src/mstar
COPY tools/distributed-training/entrypoint.sh /usr/local/entrypoint.sh
ENTRYPOINT [ "/usr/local/entrypoint.sh" ]
