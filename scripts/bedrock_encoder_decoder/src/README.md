# Setup

These scripts assume you are running the docker image built in the parent directory. This install the requirements file.

Some subfolders (i.e. benchmarking) have additional requirements that need to be installed.

# Running

The entry point is `pretrain_main.py`. This reads configurations from the hierarhical config in `config/base.yaml` using the [hydra package](hydra.cc/)

Therefore you can launch (locally) using
```
python pretrain_main.py
```

Example scripts that show overrides of config values are given in `src/scripts/sample_run.sh` and `src/scripts/mtl_sample_run.sh`


# Notes
The pytorch lightning datamodule is in `data/datamodule.py`

The pytorch lightning modulemodule is in `models/model_module.py`

## Logging

By default this uses the mstar logger and logs to mlflow. You should edit the `run_name` and `experiment_name` in `config/base.yaml`.

## Saving
The current default root directory prefix includes `/mnt_out/colehawk/easel`. This is where your checkpoints will save. Please edit the default root directory to save elsewhere. You can find this in the configuration file `config/trainer/base.yaml`.

## Learning Rate Scheduler Override

Sometimes you may want to override the learning rate scheduler. 

You can use hydra to override values in `config/optimization/scheduler.yaml`, which affects the scheduler that is instantiated, or `config/optimization/override.yaml`, which contains args for scalar multiplication and shifting the global step index. See [link](https://gitlab.aws.dev/mstar/mstar/-/blob/master/scripts/bedrock_encoder_decoder/src/models/modelmodule.py#L86) for the exact implementation of shifting and scalar multiplcation.

The global step is fed to the scheduler, and neither the global step nor the optimizer learning rate can be modified without restarting a run since they are overwritten by deepspeed on checkpoint resume. 


## Hydra Help

The best starting point is [the hydra docs](https://hydra.cc/docs/intro/)

To override a config group use `/` not `.`. If you use `.` this will assign a value not a config group.

For example 
```
#good, use / to assign a group
python pretrain_main.py optimization/scheduler=linear
#good, use . to assign a value
python pretrain_main.py optimization.scheduler.num_warmup_steps=100 


#will fail
python pretrain_main.py optimization.scheduler=linear
```
