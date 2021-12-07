import os
import json
import time

from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.callbacks.progress import ProgressBarBase


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

    def on_train_batch_end(  # pylint: disable=unused-argument
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
                f"It/s {it_per_seconds:.2f} Hours left {(self._total_steps - self._train_batch_idx)/it_per_seconds/3600:.2f}]:",
                self.trainer.progress_bar_dict,
            )
