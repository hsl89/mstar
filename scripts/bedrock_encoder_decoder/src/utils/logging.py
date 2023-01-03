"""
Utility functions for main training run logging
"""
import datetime
import random
import os
import mstar.utils.lightning
from pytorch_lightning.utilities.logger import _convert_params, _flatten_dict
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from typing import Optional,Dict, Any, Union
from argparse import Namespace

def get_save_dir(cfg):  # , mstar_logger):

    base_save_path = os.path.join(cfg.trainer.default_root_dir, cfg.run_name)

    if getattr(cfg,"save_by_timestamp",False):
        run_tag = datetime.datetime.now().strftime("%m_%d_%H_%s")
    else:
        run_tag = os.environ.get('JOB_NAME')
 
    save_dir_path = os.path.join(base_save_path, run_tag)

    return save_dir_path

class MyMStarEKSLogger(mstar.utils.lightning.MStarEKSLogger):
    """
    Override methods of mstar logger that fail when upgrading lightning
    """
    def __init__(
        self,
        experiment_name: str = os.getenv("MSTAR_TRAINING_JOB_NAME"),
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = {"mode": "training"},
        s3_upload=False,
    ):
        super().__init__(
            experiment_name=experiment_name,
            run_name=run_name,
            tags=tags,
            s3_upload=s3_upload
        )

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        """
        Override since rank_zero_warn changes behavior in lightning 1.8.6
        """
        params = _convert_params(params)
        params = _flatten_dict(params)
        for k, v in params.items():
            if len(str(v)) > 250:
                rank_zero_warn(
                    f"Mlflow only allows parameters with up to 250 characters. Discard {k}={v}"
                )
                continue

            self.experiment.log_param(self.run_id, k, v)
