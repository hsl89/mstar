# MStar - Decoder Model Pretraining

Source code for pretraining transformer based large scale decoder language models similar to GPT.

## Docker Build
The following is an example docker build command on EC2 and push it to us-east-1
```
bash docker_build.sh ${REGION} ${TAG}
```

## Start training on MStar-EKS cluster
Some example config files to train decoder models can be found in `example_eks_config`. Please update the docker image with your own docker image created by following the instructions above. You should also update the model architecture and related hyper-parameters.

```
mstarxp submit -f example_eks_config/config_bedrock_26B_input2048_mstar_t5_tokenizer_tokens215B_batch1.32M_lr2e-4_p4d.yml
```
