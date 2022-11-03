Please read this first: https://quip-amazon.com/mb4BAGjU3icv/M-EKS-Tutorial

The linked onboarding guide covers the MStar cluster.

## Docker build (EC2) 

EC2 is faster for both docker builds and pushes (higher bandwith than local wifi). You can (1) install docker on your EC2 dev machine (2) authenticate using the gluonnlp temporary credentials, and then (3) change `NAME` and `TAG` in `scripts/ec2_all_region_docker` and finally (4) use 
```
bash scripts/ec2_all_region_docker.sh
```
to build the dev and non-dev dockerfiles and push to all regions. See `scripts/ec2_all_region_docker` for more information. 

## Docker build (Local)

You can use
```
bash scripts/docker.sh $REGION $TAG
```
to build the `Dockerfile` and push a docker image tagged `$(whoami)-$TAG` to the `mstar-eks` ECR in region `REGION`.

## Additional Info (Can Skip)

The docker scripts copy over the current directory. Also, they pull from the mstar master branch docker image in the Elastic Container Repository `mstar-gitlab:master` in region `us-east-2` (of the gluonnlp account). This is only up-to-date on the first pull. After that your local machine will cache the image, so you can run 
```
docker system prune -a
```
to clear out your local cache. Then rebuilding the container will put you up-to-date with the MStar master branch. Otherwise your build will always use the local cache and it will be out-of-date.

Note that this is different than the gitlab setup described in the EKS tutorial linked above. 


# Dev workspace setup
This assumes that you will create a dev workspace on the MStar EKS cluster.

This assumes you have already created a development dockerfile. You can also build an image from the `dev_Dockerfile` by editing `scripts/ec2_all_region_docker.sh` so that `DEVS=('0' '1')`. Adding the `1` flag also builds from the `dev_Dockerfile` and pushes dev images in each region.
The dev images add some utilities (vim, tmux, git) in addition to installing requirements and copying over the code.

# Launching a job

You can launch using the configs in `configs`.


# Known Issues

Things that can cause issues that are not easy to diagnose. If you run into an issue, please submit a PR!

## Sharded Context for Model Creation

Currently only the loading method `safe_state_dict` initializes the model in the sharded context. Recommended whenever using Zero-2D/Zero-3.


## Deepspeed Sharding+Val Sanity Steps

When using Zero-2D/Zero-3 set `num_val_sanity_steps=0`. Otherwise you will get an error like
``` 
RuntimeError: tracing error
```

## Offline Example Packing

The collators rely on the assumption of offline example packing. They are not guaranteed to handle padding well.


