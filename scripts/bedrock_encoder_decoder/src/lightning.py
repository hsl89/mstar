import os
import mstar
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


class KubeFlowEnvironment(LightningEnvironment):
    def __init__(self, main_port: int):
        super().__init__()
        self._main_port = main_port

    """ 
    @property 
    def main_port(self) -> str:
        return os.environ["MASTER_ADDR"]
    """

    @property
    def main_address(self) -> str:
        return os.environ["MASTER_ADDR"]

    @property
    def main_port(self) -> int:
        if "MASTER_PORT" in os.environ:
            # Lightning may spawn processes and set MASTER_PORT in their env
            assert int(os.environ["MASTER_PORT"]) == self._main_port
        return self._main_port

    def node_rank(self) -> int:
        return int(os.environ["RANK"])

    def world_size(self) -> int:
        return int(os.environ["REAL_WORLD_SIZE"])


class MyMStarEKSLogger(mstar.utils.lightning.MStarEKSLogger):
    def __init__(
        self,
        experiment_name: str = os.getenv("MSTAR_TRAINING_JOB_NAME"),
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = {"mode": "training"},
        do_s3_upload: bool = False,
    ):
        super().__init__()

        self._experiment_name = experiment_name
        self._experiment_id = None
        self._run_name = run_name
        self._run_id = None
        tags["job_id"] = os.environ.get("JOB_NAME", None)
        self.tags = tags
        self.do_s3_upload = do_s3_upload
        self._mlflow_client = MlflowClient(TRACKING_URI)

    @rank_zero_only
    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint):
        if self.do_s3_upload and os.path.exists(checkpoint_callback.best_model_path):
            try:
                self.experiment.log_artifact(
                    self.run_id, checkpoint_callback.best_model_path
                )
            except Exception as e:
                logger.warning("Failed to upload checkpoints...")
        else:
            logger.warning("Skipping s3 upload")


class MyAWSBatchProgressBar(ProgressBarBase):
    def __init__(self, refresh_rate: int = 25, total_steps: int = 250000):
        super().__init__()
        self._active = True
        self._train_epoch_idx = 0
        self._refresh_rate = refresh_rate
        self._last_batch_end_logged = None
        self._total_steps = total_steps
        self._train_batch_idx = 0
        self._train_epoch_idx = 0

    def disable(self):
        self._active = False

    def enable(self):
        self._active = True

    def print(self, *args, **kwargs):
        if self._active:
            print(*args, **kwargs)

    def on_train_epoch_start(self, trainer, pl_module):
        self._train_epoch_idx += 1
        self._last_batch_end_logged = time.time()

    def on_train_batch_end(  # pylint: disable=unused-argument, arguments-renamed
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        self._train_batch_idx += 1
        if self._train_batch_idx % self._refresh_rate == 0:
            it_per_seconds = self._refresh_rate / (
                time.time() - self._last_batch_end_logged
            )
            self._last_batch_end_logged = time.time()
            self.print(
                f"[Epoch {self._train_epoch_idx} Batch {self._train_batch_idx} "
                f"It/s {it_per_seconds:.3f} Hours left {(self._total_steps - self._train_batch_idx)/it_per_seconds/3600:.2f}]:",
                self.trainer.progress_bar_dict,
            )
