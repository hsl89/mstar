import pytorch_lightning as pl
import torch as th
import os,sys
import hydra 
from omegaconf import DictConfig, OmegaConf
import logging
from utils.eks_utils import eks_setup
from typing import List, Optional
from pytorch_lightning import Callback
from pytorch_lightning.loggers import TensorBoardLogger

@hydra.main(config_path="conf", config_name="config_legal")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    working_dir = os.getcwd()
    print("Working directory : {}".format(working_dir))
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    model_args = cfg.model
    data_args = cfg.datamodule
    trainer_args = cfg.training
    optimizer_cfg = cfg.model.OptimizerArgs
    eks_args = cfg.EKSArgs
      
    logger.info(f"model_args: {model_args}\n\n")
    logger.info(f"data_args: {data_args}\n\n")
    logger.info(f"trainer_args: {trainer_args}\n\n")

    pl.utilities.seed.seed_everything(optimizer_cfg.seed)
    
    # EKS boilerplate code
    is_on_eks_cluster = False
    plugins = []
    is_on_eks_cluster, plugins, mstar_logger = eks_setup(plugins, 
                                                         is_on_eks_cluster, 
                                                         trainer_args, 
                                                         eks_args,
                                                         cfg)  

    # Add plugins.
    if "plugins" in model_args:
        for _, cb_conf in model_args.plugins.items():
            if "_target_" in cb_conf:
                logger.info(f"Instantiating plugin <{cb_conf._target_}>")
                plugins.append(hydra.utils.instantiate(cb_conf))
    
    # Init tokenizer
    tokenizer_type = model_args.tokenizer_type
    tokenizer_class = hydra.utils.get_class(model_args.tokenizer_class)
    tokenizer = tokenizer_class.from_pretrained(tokenizer_type)

    # Init Model Module
    model_class = hydra.utils.get_class(model_args.model_class)
    model = model_class.from_pretrained(model_args.model_type)
    plmodule = hydra.utils.get_class(model_args.pl_modelmodule)
    model_module = plmodule(model, tokenizer, logger, model_args, data_args)    

    # Init Data Module
    pldata_class = hydra.utils.get_class(data_args.datamodule_class)
    data_module = pldata_class(tokenizer, logger, data_args)
    
    # Training 
    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in model_args:
        for _, cb_conf in model_args.callbacks.items():
            if "_target_" in cb_conf:
                logger.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))
    
    trainer_logger = mstar_logger if is_on_eks_cluster else TensorBoardLogger(working_dir)
    if trainer_args.gpus == -1:
        trainer_args.gpus = th.cuda.device_count()
    
    trainer = pl.Trainer(**trainer_args,
                        logger=trainer_logger,
                        callbacks=callbacks,
                        plugins=plugins)
    
    if is_on_eks_cluster:
        # Those two steps need to set after trainer initialization. 
        user_log_dict = { "working_dir": working_dir}
        mstar_logger.log_env_as_artifact(user_log_dict)
        mstar_logger.log_hyperparams(cfg)
    if cfg.train_mode:
        trainer.fit(model_module, data_module)        

    ## Test
    if cfg.train_mode:
        assert isinstance(callbacks[0], pl.callbacks.ModelCheckpoint)
        logger.info(f"Best ckpt path: {callbacks[0].best_model_path}")
        trainer.test(ckpt_path=callbacks[0].best_model_path, 
                    dataloaders=data_module.test_dataloader())
    else:
        trainer.test(model=model_module, dataloaders=data_module.test_dataloader())


if __name__ == "__main__":
    main()
