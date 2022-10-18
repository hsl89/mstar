#!/bin/bash
TORCH_CUDA_ARCH_LIST="8.0 7.5 7.0" DS_BUILD_UTILS=1 DS_BUILD_FUSED_LAMB=0 python3 -m pip install DeepSpeed-ZeRO-2D.tar.gz --no-build-isolation
