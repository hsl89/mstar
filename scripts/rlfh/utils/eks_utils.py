import os
import json
import time
import logging
import re

from argparse import Namespace

from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.callbacks.progress import ProgressBarBase
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from pytorch_lightning.utilities.logger import _convert_params, _flatten_dict
from mlflow.tracking import context, MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
from src.mstar.utils.lightning import MStarEKSLogger, KubeFlowEnvironment

from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Tracking URI for mlflow tracking server in EKS
TRACKING_URI = "http://mlflow-tracking-server.mlflow.svc.cluster.local:80"

if hasattr(context, "registry"):
    from mlflow.tracking.context.registry import resolve_tags
else:
    def resolve_tags(tags=None):
        return tags

    
def eks_setup(plugins, is_on_eks_cluster, trainer_args, eks_args, cfg):
    mstar_logger = None
    if "KUBERNETES_SERVICE_HOST" in os.environ:
        is_on_eks_cluster = True
        num_nodes = trainer_args.num_nodes
        kubeflow_num_nodes = int(os.environ["NUM_NODES"])
        if num_nodes != kubeflow_num_nodes:
            logging.warning(
                f"--trainer.num_nodes={num_nodes} != "
                f"$NUM_NODES={kubeflow_num_nodes}. "
                f"Setting --trainer.num_nodes={kubeflow_num_nodes}!"
            )
            trainer_args.num_nodes = kubeflow_num_nodes
        
        mstar_logger = MStarEKSLogger(experiment_name=eks_args.experiment_name,
                              run_name=eks_args.run_name,
                              tags={"mode": "Training"})
        plugins.append(KubeFlowEnvironment(master_port=23456)) 
    return is_on_eks_cluster, plugins, mstar_logger