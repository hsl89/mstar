# Setup

These scripts assume you are running the docker image built in the parent directory. This install the requirements file.

Some subfolders (i.e. benchmarking) have additional requirements that need to be installed.

# Running

The entry point is `pretrain_main.py`. This reads configurations from the hierarhical confi in `config/base.yaml` using the [hydra package](hydra.cc/)

Therefore you can launch (locally) using
```
python pretrain_main.py
```
and override arguments as in `scripts/test_launch.sh`.

Please read through the hydra documentation linked above for more examples.

# Notes
The pytorch lightning datamodule is in `data/datamodule.py`

The pytorch lightning modulemodule is in `models/model_module.py`

## Logging

By default this uses the mstar logger and logs to mlflow. You should edit the `run_name` and `experiment_name` in `config/base.yaml`.

## Saving
The current default root directory prefix includes `/mnt_out/colehawk/easel`. This is where your checkpoints will save. Please edit the default root directory to save elsewhere. You can find this in the configuration file `config/trainer/base.yaml`.
