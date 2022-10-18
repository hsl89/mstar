#!/bin/bash
#install AWS CLI for s3 work
cd /tmp
apt-get update
apt-get install -y curl
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && unzip awscliv2.zip && ./aws/install
alias aws='/usr/local/bin/aws'
