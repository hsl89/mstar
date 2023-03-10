Please read this first: https://quip-amazon.com/mb4BAGjU3icv/M-EKS-Tutorial

The linked onboarding guide covers the MStar cluster.

## Docker build (EC2) 

EC2 is faster for both docker builds and pushes (higher bandwith than local wifi). You can (1) install docker on your EC2 dev machine (2) authenticate using the gluonnlp temporary credentials, and then (3) use
```
bash scripts/ec2_all_region_docker.sh $USER $CONTAINER_DESCRIPTION
```
to build docker images and push to all regions. See `scripts/ec2_all_region_docker.sh` for more information. The actual docker build is performed in `scripts/ec2_build_docker.sh` and takes places from the parent directory of `mstar`. 

## Docker build (Local)

You can use
```
bash scripts/docker.sh $REGION $TAG
```
to build the `Dockerfile` and push a docker image tagged `$(whoami)-$TAG` to the `mstar-eks` ECR in region `REGION`.

## Sample launch command

`bash scripts/sample_run.sh`

## Additional Info (Can Skip)

The docker scripts copy over the current directory. Also, they pull from the mstar master branch docker image in the Elastic Container Repository `mstar-gitlab:master` in region `us-east-2` (of the gluonnlp account). This is only up-to-date on the first pull. After that your local machine will cache the image, so you can run 
```
docker system prune -a
```
to clear out your local cache. Then rebuilding the container will put you up-to-date with the MStar master branch. Otherwise your build will always use the local cache and it will be out-of-date.

Note that this is different than the gitlab setup described in the EKS tutorial linked above. 

# Launching a job

You can launch using the configs in `configs`.

# Auto-restart (Temporary feature)

Instead of passing `model.ckpt_path` to resume, you can pass `model.ckpt_path='auto`. This will search for the latest checkpoint in the run directory.

Also, you can filter out keywords by adding to the hydra config. It's necessary to double-escape the "=" for json parsing during mstarx job submission. Normal hydra parsing would require only one escape. It's recommended to 
```
#filter out last.ckpt, recommended
++filter_keywords=['last.ckpt']

#filter out checkpoints containing `last.ckpt` or checkpoints that contain `step=200` or `step=150`
++filter_keywords=['last.ckpt','step\\=200','step\\=150']
```

Run names should be different from previous runs, otherwise the deepspeed checkpoint from the previous run will be loaded. This is particularly important for Stage 2 configs, since the deepspeed checkpoint will overwrite the state dict even if a packaged model is provided.


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


