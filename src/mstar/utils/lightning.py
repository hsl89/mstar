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


class AWSBatchEnvironment(LightningEnvironment):
    def __init__(self, master_port: int):
        super().__init__()
        self._master_port = master_port

    def master_address(self) -> str:
        # AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS isn't present on the main node
        return os.environ.get(
            "AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS", "127.0.0.1"
        )

    def master_port(self) -> int:
        if "MASTER_PORT" in os.environ:
            # Lightning may spawn processes and set MASTER_PORT in their env
            assert int(os.environ["MASTER_PORT"]) == self._master_port
        return self._master_port

    def node_rank(self) -> int:
        return int(os.environ["AWS_BATCH_JOB_NODE_INDEX"])


class SageMakerEnvironment(LightningEnvironment):
    def __init__(self, master_port: int):
        super().__init__()
        self._master_port = master_port

    def master_address(self) -> str:
        sm_resource_file = "/opt/ml/input/config/resourceconfig.json"
        with open(sm_resource_file, "r", encoding='utf-8') as f:
            sm_resources = json.load(f)
            return sm_resources["hosts"][0]

    def master_port(self) -> int:
        if "MASTER_PORT" in os.environ:
            # Lightning may spawn processes and set MASTER_PORT in their env
            assert int(os.environ["MASTER_PORT"]) == self._master_port
        return self._master_port

    def node_rank(self) -> int:
        sm_resource_file = "/opt/ml/input/config/resourceconfig.json"
        with open(sm_resource_file, "r", encoding='utf-8') as f:
            sm_resources = json.load(f)
        return sm_resources["hosts"].index(sm_resources["current_host"])


class KubeFlowEnvironment(LightningEnvironment):
    def __init__(self, master_port: int):
        super().__init__()
        self._master_port = master_port

    def master_address(self) -> str:
        return os.environ["MASTER_ADDR"]

    #needs to be property for PTL 1.8 
    @property
    def main_address(self) -> str:
        return self.master_address()

    def master_port(self) -> int:
        if "MASTER_PORT" in os.environ:
            # Lightning may spawn processes and set MASTER_PORT in their env
            assert int(os.environ["MASTER_PORT"]) == self._master_port
        return self._master_port

    #needs to be property for PTL 1.8 
    @property
    def main_port(self) -> int:
        return self.master_port()

    def node_rank(self) -> int:
        return int(os.environ["RANK"])

    def world_size(self) -> int:
        return int(os.environ["REAL_WORLD_SIZE"])


class AWSBatchProgressBar(ProgressBarBase):
    def __init__(self, refresh_rate: int = 25, total_steps: int = 250000):
        super().__init__()
        self._active = True
        self._train_epoch_idx = 0
        self._refresh_rate = refresh_rate
        self._last_batch_end_logged = None
        self._total_steps = total_steps

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
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if self.train_batch_idx % self._refresh_rate == 0:
            it_per_seconds = self._refresh_rate / (
                time.time() - self._last_batch_end_logged
            )
            self._last_batch_end_logged = time.time()
            self.print(
                f"[Epoch {self._train_epoch_idx} Batch {self.train_batch_idx} "
                f"It/s {it_per_seconds:.3f} Hours left {(self._total_steps - self.train_batch_idx)/it_per_seconds/3600:.2f}]:",
                self.trainer.progress_bar_dict,
            )


class MStarEKSLogger(LightningLoggerBase):
    def __init__(
        self,
        experiment_name: str = os.getenv("MSTAR_TRAINING_JOB_NAME"),
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = {"mode": "training"},
        s3_upload=False,
    ):
        super().__init__()

        self._experiment_name = experiment_name
        self._experiment_id = None
        self._run_name = run_name
        self._run_id = None
        tags['job_id'] = os.environ.get('JOB_NAME', None)
        self.tags = tags
        self.s3_upload = s3_upload

        self._mlflow_client = MlflowClient(TRACKING_URI)

    @rank_zero_experiment
    def get_client(self) -> MlflowClient:
        if self._experiment_id is None:
            if self._experiment_name is None:
                self._experiment_name = 'default'
            experiment = self._mlflow_client.get_experiment_by_name(self._experiment_name)
            if experiment is not None:
                self._experiment_id = experiment.experiment_id
            else:
                self._experiment_id = self._mlflow_client.create_experiment(self._experiment_name)

        if self._run_id is None:
            if self._run_name is not None:
                self.tags[MLFLOW_RUN_NAME] = self._run_name
            run = self._mlflow_client.create_run(experiment_id=self._experiment_id, tags=resolve_tags(self.tags))
            self._run_id = run.info.run_id
        return self._mlflow_client

    @property
    @rank_zero_experiment
    def experiment(self) -> MlflowClient:
        return self.get_client()

    @property
    def run_id(self) -> str:
        self.get_client()
        return self._run_id

    @property
    def experiment_id(self) -> str:
        self.get_client()
        return self._experiment_id

    @property
    def name(self) -> str:
        return self.experiment_id

    @property
    def version(self) -> str:
        return self.run_id

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = _convert_params(params)
        params = _flatten_dict(params)
        for k, v in params.items():
            if len(str(v)) > 250:
                rank_zero_warn(
                    f"Mlflow only allows parameters with up to 250 characters. Discard {k}={v}", RuntimeWarning
                )
                continue

            self.experiment.log_param(self.run_id, k, v)


    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        timestamp_ms = int(time.time() * 1000)
        for k, v in metrics.items():
            if isinstance(v, str):
                logger.warning("Discarding metric with string value {0}={1}." % k, v)
                continue

            new_k = re.sub("[^a-zA-Z0-9_/. -]+", "", k)
            if k != new_k:
                rank_zero_warn(
                    "MLFlow only allows '_', '/', '.' and ' ' special characters in metric name."
                    f" Replacing {k} with {new_k}.",
                    RuntimeWarning,
                )
                k = new_k

            self.experiment.log_metric(self.run_id, k, v, timestamp_ms, step)

    @rank_zero_only
    def log_artifact(self, file_path: str) -> None:
        self.experiment.log_artifact(self.run_id, file_path)

    @rank_zero_only
    def log_dict(self, dictionary: Any, artifact_file: str) -> None:
        self.experiment.log_dict(self.run_id, dictionary, artifact_file)

    @rank_zero_only
    def finalize(self, status: str = "FINISHED") -> None:
        super().finalize(status)
        status = "FINISHED" if status == "success" else status
        if self.experiment.get_run(self.run_id):
            self.experiment.set_terminated(self.run_id, status)

    @rank_zero_only
    def log_env_as_artifact(self, user_log_dict: Optional[Dict] = None) -> None: # pylint: disable=too-many-statements
        import urllib.request
        cur_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
        try:
            job_name = os.environ['JOB_NAME']
            experiment_info = {}
            job_info_path = '/mnt_out/job_store/' + job_name
            experiment_info['experiment_id'] = self._experiment_id
            experiment_info['run_id'] = self.run_id
            with open(job_info_path + '/experiment_info.json', 'w') as f:
                json.dump(experiment_info, f)
            logger.info("Logging job spec as artifact...")
            self.log_artifact('/mnt_out/job_store/' + job_name)
        except Exception as e:
            logger.warning("Failed to log job spec...")

        try:
            env_json = cur_path + '/internal_env_variables.json'
            logger.info("Logging internal environment variables as artifact...")
            with open(env_json, 'w') as f:
                json.dump(dict(os.environ), f)
            self.log_artifact(env_json)
        except Exception as e:
            logger.warning("Failed to log internal environment variables...")

        try:
            other_meta_data = cur_path + '/internal_meta_data.json'
            meta_data = {}
            with urllib.request.urlopen('http://169.254.169.254/latest/meta-data/instance-id') as f:
                meta_data['instance-id'] = f.read().decode()

            with urllib.request.urlopen('http://169.254.169.254/latest/meta-data/ami-id') as f:
                meta_data['instance-type'] = f.read().decode()

            with urllib.request.urlopen('http://169.254.169.254/latest/meta-data/placement/availability-zone') as f:
                meta_data['placement/availability-zone'] = f.read().decode()
            
            with urllib.request.urlopen('http://169.254.169.254/latest/meta-data/ami-id') as f:
                meta_data['ami-id'] = f.read().decode()

            logger.info("Logging internal metadata as artifact...")
            with open(other_meta_data, 'w') as f:
                json.dump(meta_data, f)
            self.log_artifact(other_meta_data)
        except Exception as e:
            logger.warning("Failed to log internal metadata...")
        
        if user_log_dict is not None:
            try:
                user_input_json = cur_path + '/user_meta_data.json'
                meta_data = user_log_dict
                logger.info("Logging user metadata as artifact...")
                with open(user_input_json, 'w') as f:
                    json.dump(meta_data, f)
                self.log_artifact(user_input_json)
            except Exception as e:
                logger.warning("Failed to log user metadata...")

        try:
            git_info_file = cur_path + '/git_info.json'
            git_info = {}
            import git
            repo = git.Repo(search_parent_directories=True)
            git_info['sha'] = repo.head.object.hexsha
            logger.info("Logging git info as artifact...")
            with open(git_info_file, 'w') as f:
                json.dump(git_info, f)
            self.log_artifact(git_info_file)
        except Exception as e:
            logger.warning("Failed to log git info...")

        try:
            package_version_file = cur_path + '/package_version.txt'
            logger.info("Logging package version info as artifact...")
            os.system("pip list --format=freeze > {}".format(package_version_file))
            self.log_artifact(package_version_file)
        except Exception as e:
            logger.warning("Failed to package version info...")

    @rank_zero_only
    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint):
        if self.s3_upload:
            if os.path.exists(checkpoint_callback.best_model_path):
                try:
                    self.experiment.log_artifact(self.run_id, checkpoint_callback.best_model_path)
                except Exception as e:
                    logger.warning("Failed to upload checkpoints...")
