#!/bin/bash

#see explanation of nsys options here
#https://gist.github.com/mcarilli/376821aa1a7182dfcf59928a7cde3223

nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu -o nsight/nsight_report -f true --capture-range=cudaProfilerApi --cudabacktrace true -x true python3 tflops_improvement.py 
