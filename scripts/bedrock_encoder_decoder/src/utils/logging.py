"""
Utility functions for main training run logging
"""
import datetime
import random
import os


def get_save_dir(cfg):  # , mstar_logger):

    base_save_path = os.path.join(cfg.trainer.default_root_dir, cfg.run_name)

    if getattr(cfg,"save_by_timestamp",False):
        run_tag = datetime.datetime.now().strftime("%m_%d_%H_%s")
    else:
        run_tag = os.environ.get('JOB_NAME')
 
    save_dir_path = os.path.join(base_save_path, run_tag)

    return save_dir_path
