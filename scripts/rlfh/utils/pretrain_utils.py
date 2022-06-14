from typing import Dict, List, Tuple
import torch
import torchmetrics 
import numpy as np
from tempfile import mkdtemp, TemporaryFile
import os.path as path
import os


class LossMetric(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            "loss", default=torch.tensor([0], dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.add_state(
            "count",
            default=torch.tensor([0], dtype=torch.float64),
            dist_reduce_fx="sum",
        )

    def update(self, loss: torch.Tensor, count: torch.Tensor):
        self.loss += loss.double() * count.double()
        self.count += count.double()

    def compute(self):
        return self.loss / self.count
