#!/bin/bash

#tmp dir for install
cd /tmp

#download from s3
aws s3 cp s3://colehawk/NsightSystems-linux-public-2022.2.1.31-5fe97ab.run .

#make executable
chmod +x NsightSystems-linux-public-2022.2.1.31-5fe97ab.run

#run (and accept)
./NsightSystems-linux-public-2022.2.1.31-5fe97ab.run

#add to path
echo "export PATH="/opt/nvidia/nsight-systems/2022.2.1/bin:${PATH}"" >> ~/.bashrct PATH="/opt/nvidia/nsight-systems/2022.2.1/bin:${PATH}" >> ~/.bashrc

#used for annotations
python -m pip install nvtx

