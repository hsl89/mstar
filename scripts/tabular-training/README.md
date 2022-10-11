Please read this first: https://quip-amazon.com/mb4BAGjU3icv/M-EKS-Tutorial

The linked onboarding guide covers the MStar cluster, but this folder provides some additional cluster launch/docker build utilities.

# Docker build/launch

This is setup for running on the EKS cluster, so you need a docker container.

The script 
```
bash scripts/docker.sh $REGION $TAG 
```
will build a dockerfile in the specified region. A sample build command is 
```
bash scripts/docker.sh us-east-1 easel  
```
See `scripts/docker.sh` for more arg information. Note that passing the flag `1` is for the dev container and clones your dotfiles, which will fail if you don't have dotfiles.

The tag affects the name of your dockerfile in the ECR. Your username is automatically inferred by `whoami`.

The command above will build and push a container called `$(whoami)-easel` to `us-east-1` Elastic Container Registry in the gluonnlp account. See the MStar EKS onboarding guide for more information. 

## Additional Info (Can Skip)

The docker scripts copy over the current directory. Also, they pull from the mstar master branch docker image in the Elastic Container Repository (of the gluonnlp account). This is only up-to-date on the first pull. After that your local machine will cache the image, so you can run 
```
docker system prune -a
```
to clear out your local cache. Then rebuilding the container will put you up-to-date with the MStar master branch. Otherwise your build will always use the local cache and it will be out-of-date.

Note that this is different than the gitlab setup described in the EKS tutorial linked above. 

## EC2 Docker build (Can Skip)

EC2 is faster for docker builds and pushes (higher bandwith than local wifi). You can (1) install docker on your EC2 dev machine (2) authenticate using the gluonnlp Isengard credentials, and then (3)change `NAME` and `TAG` in `scripts/ec2_all_region_docker` and finally (4) use 
```
bash scripts/ec2_all_region_docker.sh
```
to build the dev and non-dev dockerfiles and push to all regions. See `scripts/ec2_all_region_docker` for more information. Notice that this is a different build script because we need to add the user name.

Your EC2 username is likely `ubuntu` or `ec2-user` and you need to change the name/docker tag. Edit `scripts/ec2_all_region_docker.sh` to change the tag and the name.

# Dev workspace setup

This assumes you have already created a development dockerfile.

This assumes that you will create a dev workspace on the MStar EKS cluster. The Dockerfile for the dev workspace is `dev_Dockerfile` which adds some utilities (vim, tmux, git) in addition to installing requirements and copying over the code.

Everything is set up to be launched from `scripts/launch.py`. You can create a `dev_Dockerfile` that will be used to build this workspace. Right now there is no `dev_Dockerfile` in this repo. You can copy the `Dockerfile` and add utilities (i.e. vim, tmux, git).

After that you can launch a workspace with 
```python
python scripts/launch.py --region REGION --type workspace
```
This will launch a single-node workspace. You can change many workspace attributes using the args of `scripts/launch.py`. You should test that your launch commands work here.

# Launching a job

You can launch using the `configs/pretrain_launch.yaml` file using 
```python
python scripts/launch.py --region REGION --type pretrain
```
Note that this does not automatically rename your job. You may want to edit the config file directly to change `name`.


# Index Files and Launching

You must generate index files with `bash src/scripts/indices/generate_idx_files.sh` before launching. You will need to put those into `mstar-data` (see EKS tutorial). You will need to edit the script based on the number of nodes/gpus you plan to use. The recommended use is pre-generating index files for common node numbers (8,16,32) and then editing the index file path using Hydra at runtime. See `src/README.md` for more Hydra info.

WARNING: CURRENTLY HARDCODED to 8 nodes, 8 gpus. You need to adjust the configuration in `src/config/data/base.yaml` after generating new index files. You can modify this at runtime using Hydra.

The data is read from `/mnt/<path>` which mirrors the s3 path `mstar-data/<path>`. It's fine to read from here, just please don't write. Also this data source is not guaranteed to be stable, though it is not fast-moving. 

# Custom Model

This repo loads a custom T5 model that is very similar to the huggingface T5 model. The code is in `models/t5_model.py`. 

The custom model incorporates the megatron fused softmax through `mstar.megatron`. We can test for equal outputs given equal weights, but the gradients are expected to be different. You can test for equal outputs using 
```
python test_model.py --softmax-type mstar_fused --use-fused-attention.
```
Numerical differences in the gradient are expected based on the megatron repo, so we print these as well, but unequal gradients do not constitute a "failed" model so training should be fine. We use the fused softmax in training. We will move this model to the mstar model factory.

# Resuming

Use the `ckpt_path` argument in `config/trainer/base.yaml`. This points to a checkpoint directory in `mnt_out` that you have already specified. For example `mnt_out/colehawk/easel/run_10/step_50000_train_loss_0.918.ckpt`. This resumes with both optimizer and model states. Deepspeed resume requries that the number of gpus is constant across restarts. 

## Resume+Errorbar/Logging

Resuming leads to negative time estimates/wrong step counts. This does not affect training correctness.

See https://github.com/PyTorchLightning/pytorch-lightning/issues/13124
