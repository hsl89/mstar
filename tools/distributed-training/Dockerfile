#syntax=docker/dockerfile:1.2
FROM nvidia/cuda:11.4.0-devel-ubuntu20.04 as base
LABEL maintainer="MStar Team"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    SHELL=/bin/bash

RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y \
        build-essential autoconf libtool cmake ninja-build fuse iproute2 \
        libcudnn8 libcudnn8-dev \
        libzstd-dev wget git unzip python3-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Require ccache>=3.7.9 to cache nvcc outputs
RUN cd /usr/local/src && \
    git clone --recursive https://github.com/ccache/ccache.git && \
    cd ccache && \
    git checkout v4.3 && \
    mkdir build; cd build; cmake -GNinja -DCMAKE_BUILD_TYPE=Release .. && \
    ninja && ninja install && \
    cd /usr/local/src && \
    ln -s /usr/local/bin/ccache /usr/local/bin/nvcc && \
    rm -rf ccache

# EFA Support
ENV LD_LIBRARY_PATH=/usr/local/src/nccl/build/lib:/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:$PATH
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y nvidia-fabricmanager-470 && \
    apt-get purge --allow-change-held-packages -y libnccl2 libnccl-dev && \
    cd /usr/local/src && \
    wget https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz && \
    tar -xf aws-efa-installer-latest.tar.gz && \
    cd aws-efa-installer && \
    ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify && \
    cd /usr/local/src && rm -rf aws-efa-installer-latest.tar.gz aws-efa-installer && \
    # Only claims support for nccl v2.9.9-1; Greg Inozemtsev confirmed v2.10 also works, which is needed for bfloat16
    cd /usr/local/src && git clone -b v2.10.3-1 https://github.com/NVIDIA/nccl.git && cd nccl && \
    make -j src.build CUDA_HOME=/usr/local/cuda \
      NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_75,code=sm_75" && \
      make install && \
    git clone -b v1.1.3-aws https://github.com/aws/aws-ofi-nccl.git /opt/aws-ofi-nccl && \
    cd /opt/aws-ofi-nccl && \
    ./autogen.sh && \
    ./configure --prefix=/opt/aws-ofi-nccl/install \
      --with-libfabric=/opt/amazon/efa/ \
      --with-cuda=/usr/local/cuda \
      --with-nccl=/usr/ \
      --with-mpi=/opt/amazon/openmpi/ && \
    make && make install && \
    rm -rf /var/lib/apt/lists/*

# # LD_PRELOAD_TRICK for pypi torch
# ENV LD_PRELOAD=/usr/local/src/nccl/build/lib/libnccl.so
# RUN --mount=type=cache,target=/root/.cache/pip \
#     python3 -m pip install awscli pyarrow scikit-build && \
#     pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html && \
#     # python3 -m pip install torchvision --no-dependencies && \
#     python3 -m pip install fairscale --no-build-isolation && \
#     TORCH_CUDA_ARCH_LIST="8.0" DS_BUILD_UTILS=1 DS_BUILD_FUSED_LAMB=1 python3 -m pip install deepspeed \
#       --no-build-isolation

RUN --mount=type=cache,target=/root/.cache/pip --mount=type=cache,target=/root/.ccache  \
    python3 -m pip install awscli pyarrow scikit-build pyyaml typing-extensions numpy && \
    cd /usr/local/src/ && \
    git clone --recursive -j$(nproc) -b release/1.9 https://github.com/pytorch/pytorch.git && \
    cd pytorch && \
    TORCH_CUDA_ARCH_LIST="8.0" USE_SYSTEM_NCCL=ON BUILD_CAFFE2_OPS=0 BUILD_CAFFE2=0 USE_MPI=0 \
        CUDA_NVCC_EXECUTABLE=/usr/local/bin/nvcc python3 setup.py install && \
    cd /usr/local/src/ && rm -rf pytorch && \
    python3 -m pip install fairscale --no-build-isolation && \
    TORCH_CUDA_ARCH_LIST="8.0" DS_BUILD_UTILS=1 DS_BUILD_FUSED_LAMB=1 python3 -m pip install deepspeed \
      --no-build-isolation

# Config needed to remove costly IPv4 NAT
# # Enable AWS S3 IPv6 Support
# # Note: AWS Batch Multi-Node jobs only have IPv6 Internet Connectivity (as we
# # don't use IPv4 NAT). See "Using a VPC in dual-stack mode" at
# # https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-networking.html
# RUN aws configure set default.s3.use_dualstack_endpoint true && \
#     aws configure set default.s3.addressing_style virtual

# Copy the source code and install 
RUN --mount=target=/mnt  --mount=type=cache,target=/root/.cache/pip\
    git clone --recursive /mnt /usr/local/src/mstar && \
    cd /usr/local/src/mstar && \
    git remote remove origin && \
    TORCH_CUDA_ARCH_LIST="8.0" FORCE_CUDA=1 python3 -m pip install /usr/local/src/mstar --no-build-isolation

WORKDIR /usr/local/src/mstar
COPY tools/distributed-training/entrypoint.sh /usr/local/entrypoint.sh
ENTRYPOINT [ "/usr/local/entrypoint.sh" ]

# NOTE: RUNNING NCCL TEST
# # SSH support for MPI (eg. nccl-tests)
# RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt \
#     apt-get update && apt-get install -y openssh-server openssh-client && \
#     ssh-keygen -t rsa -f /root/.ssh/id_rsa -N '' && \
#     cp /root/.ssh/id_rsa.pub /root/.ssh/authorized_keys && \
#     rm -rf /var/lib/apt/lists/*

# mkdir -p /var/run/ssh; service ssh start
# /opt/amazon/openmpi/bin/mpirun --allow-run-as-root -np 16 -N 8 -x FI_PROVIDER="efa" -x FI_EFA_USE_DEVICE_RDMA=1 --oversubscribe -H 10.0.2.23,10.0.2.214 --mca btl tcp,self --bind-to none -x LD_LIBRARY_PATH=/usr/local/src/nccl/build/lib:/usr/local/cuda/extras/CUPTI/lib64:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib:/opt/aws-ofi-nccl/install/lib:$LD_LIBRARY_PATH -x PATH=/opt/amazon/openmpi/bin/:/opt/amazon/efa/bin:$PATH -x NCCL_DEBUG=INFO -x NCCL_ALGO=RING -x NCCL_SOCKET_IFNAME=eth0 -x xNCCL_PROTO=SIMPLE /usr/local/src/nccl-tests/build/all_reduce_perf -b8 -e2G -f2 -n100 --mca pml ˆcm
