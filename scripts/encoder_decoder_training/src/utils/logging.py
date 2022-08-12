"""
Utility functions for main training run logging
"""
import datetime
import random
import os


def get_save_dir(cfg):  # , mstar_logger):

    base_save_path = os.path.join(cfg.trainer.default_root_dir, cfg.run_name)

    run_tag = datetime.datetime.now().strftime("%m_%d_%H_%M")
    """ 
    if mstar_logger.run_id:
        run_tag = mstar_logger.run_id
    else:
        warnings.warn()        
        #if mstar_logger.tags['job_id']:
        #may lead to overlap, but only with repeated runs on a dev node
        #run_tag = mstar_logger.tags['job_id']+run_tag	
    
    """
    save_dir_path = os.path.join(base_save_path, run_tag)

    return save_dir_path
