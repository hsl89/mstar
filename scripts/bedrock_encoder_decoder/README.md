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
bash scripts/docker.sh us-east-1 easel 0 
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

Everything is set up to be launched from `scripts/launch.py`. This uses the `dev_Dockerfile` which adds some utilities (vim, tmux, git) on top of the base `Dockerfile`.

If you lauch using `scripts/launch.py` then you will need to remove the blocks in `dev_Dockerfile` marked `@colehawk-specific install`. These clone my dotfiles. 

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

The data is read from `/mnt/colehawk/pile` which mirrors the s3 path `mstar-data/colehawk/pile`. It's fine to read from here, just please don't write. Also this data source is not guaranteed to be stable, though it is not fast-moving. 

# Custom Model

This repo loads a custom T5 model that is "equivalent" to the huggingface T5 model. The code is in `models/t5_model.py`. 

The custom model incorporates the megatron fused softmax through `mstar.megatron`. We can test for equal outputs given equal weights, but the gradients are expected to be different. You can test for equal outputs using 
```
python test_model.py --softmax-type mstar_fused --use-fused-attention.
```
Numerical differences in the gradient are expected based on the megatron repo, so we print these as well, but unequal gradients do not constitute a "failed" model so training should be fine. We use the fused softmax in training.

# Known Issues:


## Non-deterministic resume
This requires (1) more inspection of dataset indices on reload. Right now we increment with `self.resume_index` in the datamodule based on the number of steps. However this does not deterministically reproduce the behavior of the original run. For example: run 0--->100 has different metrics than running 0--->50, resume, run 51--->100. The dataloader is not the only potential cause, but the LR load is handled by PTL.

One possiblity is re-seeding based on the global step at checkpoint save/load. Right now we suspect re-seeding may be the cause.

## Resume+Errorbar/Logging

Resuming leads to negative time estimates/wrong step counts. This does not affect training correctness.

See https://github.com/PyTorchLightning/pytorch-lightning/issues/13124

## Index files

Hard-coded index files based on GPU counts are major limitation. This could be resolved by reading the number of global gpus from the KubeFlow environment and then generating indices online, or even better using the PTL DDP sampler. In general offline index file generation (see above) decreases our ability to easily launch jobs.
