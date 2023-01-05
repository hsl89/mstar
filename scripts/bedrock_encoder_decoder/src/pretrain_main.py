"""
Main script to launch pretraining
"""

# Preliminaries
import hydra
import yaml
import logging
import os
import sys
import mstar
import mstar.models.t5
from mstar.utils.lightning import KubeFlowEnvironment, MStarEKSLogger
import torch as th
import pytorch_lightning as pl
import deepspeed
import omegaconf
# local imports
import models
import data
import collators
import utils


@hydra.main(version_base=None, config_path="config", config_name="base.yaml")
def main(cfg):
    """
    Launch pretraining
    """

    #used for logging later, avoid instantiation
    cfg_as_primitive=omegaconf.OmegaConf.to_container(cfg)

    #resolve config and interpolate values now
    #TODO(@colehawk) this instantiates some values too early
    #otherwise issues later with instantiation
    omegaconf.OmegaConf.resolve(cfg)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.info("Configuration args")
    logger.info(cfg)
            

    #use this to ensure that all checkpoints have same naming
    #otherwise, first checkpoint is saved before validation and 
    #has different naming convention
    

    if cfg.model.ckpt_path:
        if cfg.model.ckpt_path=="auto":
            #will either return most recent checkpoint or None if no ckpt present
            cfg.model.ckpt_path = utils.auto_restart.latest_ckpt_wrapper_from_cfg(cfg)

            logger.info(f"Resuming from checkpoint to {cfg.model.ckpt_path}. None indicates restarting training")
            if cfg.model.state_dict_path and cfg.model.ckpt_path:
                # checkpoint will overwrite state dict load anyway
                logger.info(f"Resuming from checkpoint to {cfg.model.ckpt_path}. Overwriting state dict path {cfg.model.state_dict_path}")
                cfg.model.state_dict_path=None

    # If dataloader is already multiprocessing, skip this
    if cfg.data.num_workers > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "False"

    assert "KUBERNETES_SERVICE_HOST" in os.environ, "Only support EKS cluster"
    kubeflow_num_nodes = int(os.environ["NUM_NODES"])

    computed_total_batch_size = (
        cfg.trainer.num_nodes * cfg.optimization.micro_batch_size * th.cuda.device_count()
    )
    logging.info(
        f"Training with {cfg.trainer.num_nodes} nodes "
        f"micro-batch size {cfg.optimization.micro_batch_size} "
        f"total batch size {computed_total_batch_size}"
        f"and {th.cuda.device_count()} devices per-node"
    )
    
    # Set seed before initializing model
    pl.utilities.seed.seed_everything(cfg.optimization.seed)

    tokenizer = hydra.utils.call(cfg.tokenizer)

    # set up huggingface-style model config
    # uses custom config class for extra args
    hf_model_config = mstar.models.t5.MStarT5Config(**cfg.model)

    # validates config for general compataiblity 
    # with traning assumptions
    utils.config.validate_config(cfg, hf_model_config)

    collator = collators.solver.get_collator(
        data_args=cfg.data,
        tokenizer=tokenizer,
        decoder_start_token_id=hf_model_config.decoder_start_token_id,
    )

    # make sure that model has enough embeddings for the tokenizer
    assert hf_model_config.vocab_size >= len(tokenizer), f"Model vocab size {hf_model_config.vocab_size} too small for tokenizer vocab size {len(tokenizer)}"

    plugins = [hydra.utils.instantiate(cfg.lightning.plugins.environment)]

    trainer_logger = hydra.utils.instantiate(cfg.lightning.logger)

    save_dir_path = utils.logging.get_save_dir(cfg)

    # strategy determines distributed training
    # deepspeed strategy requires PTL plugin augmentation
    # to run with zero-2d and internal deepspeed config
    strategy = hydra.utils.instantiate(cfg.lightning.strategy)
    
    #TODO(colehawk) move to hydra instantiation of list as well
    callbacks = [
        hydra.utils.instantiate(cfg.lightning.callbacks.checkpoint,dirpath=save_dir_path),
        hydra.utils.instantiate(cfg.lightning.callbacks.lr_monitor),
        hydra.utils.instantiate(cfg.lightning.callbacks.early_stopping),
        hydra.utils.instantiate(cfg.lightning.callbacks.progress_bar)
    ] 

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        plugins=plugins,
        strategy=strategy,
        logger=trainer_logger,
    )
 
    assert len(list(filter(None,[cfg.model.state_dict_path,cfg.model.ckpt_path])))<=1, "Resume from either cfg.model.state_dict_path or cfg.model.ckpt_path not both"

    model_init_fn = lambda x: models.utils.model_init_fn(trainer=x, state_dict_path=cfg.model.state_dict_path, hf_model_config=hf_model_config)    

    if cfg.data.source=="mtl":

        VAL_LOSS_NAMES = ['labeled_val_loss', 'validation_loss']
        assert (cfg.optimization.labeled_micro_batch_size+cfg.optimization.unlabeled_micro_batch_size == cfg.optimization.micro_batch_size), 'Sum of unlabeled and labeled micro-batch-size should equal to the total micro-batch-size'
        #######****SETUP MTL DATA-MODULE******##################
        #since we shard in the datamodule, don't let PTL re-shard
        assert cfg.trainer.replace_sampler_ddp is False

        #Unlabeled data module
        if cfg.optimization.unlabeled_micro_batch_size > 0:
                unlabeled_data_module = data.datamodule.HFDataModule(
                tokenizer=tokenizer,
                training_datasets=cfg.data.training_datasets,
                validation_datasets=cfg.data.validation_datasets,
                seed=cfg.optimization.seed,
                micro_batch_size=cfg.optimization.unlabeled_micro_batch_size,
                data_args=cfg.data,
                data_collator=collator,
                py_logger=logger,
            )
        else:
            unlabeled_data_module = None
 
        data_module = hydra.utils.instantiate(
            cfg.lightning.data_module,
            tokenizer=tokenizer,
            labeled_batch=cfg.optimization.labeled_micro_batch_size,
            max_seq_length=cfg.data.max_seq_length,
            labeled_max_ip_seq_len=cfg.data.max_seq_length,
            labeled_max_op_seq_len=cfg.data.max_output_length,
            labeled_data_path=cfg.data.labeled_data_path,
            unlabeled_data_module=unlabeled_data_module,
            py_logger=logger
        )

        model_module = hydra.utils.instantiate(
                cfg.lightning.model_module,
                _recursive_=False, #otherwise hydra tries to instantiate the full config
                full_experiment_config=cfg_as_primitive,
                model_init_fn=model_init_fn,
                py_logger=logger,
                optimizer_cfg=cfg.optimization,
                unlabeled_batch_size=cfg.optimization.unlabeled_micro_batch_size,
                labeled_batch_size=cfg.optimization.labeled_micro_batch_size,
                tokenizer=tokenizer,
                val_loss_names=VAL_LOSS_NAMES
            )

    else:
        model_module = hydra.utils.instantiate(
                cfg.lightning.model_module,
                _recursive_=False, #otherwise hydra tries to instantiate the full config
                full_experiment_config=cfg,
                model_init_fn=model_init_fn,
                py_logger=logger,
                optimizer_cfg=cfg.optimization,
            )

        #since we shard in the datamodule, don't let PTL re-shard
        assert cfg.trainer.replace_sampler_ddp is False

        if cfg.data.source=="online_packed":
            logger.info("Using online packed data")
            data_module = data.datamodule.OnlineHFDataModule(
                    tokenizer=tokenizer,
                    training_datasets=cfg.data.training_datasets,
                    validation_datasets=cfg.data.validation_datasets,
                    seed=cfg.optimization.seed,
                    micro_batch_size=cfg.optimization.micro_batch_size,
                    data_args=cfg.data,
                    data_collator=collator,
                    py_logger=logger,
            )

        elif cfg.data.source=="offline_packed":
            logger.info("Using offline packed data")
            data_module = data.datamodule.HFDataModule(
                    tokenizer=tokenizer,
                    training_datasets=cfg.data.training_datasets,
                    validation_datasets=cfg.data.validation_datasets,
                    seed=cfg.optimization.seed,
                    micro_batch_size=cfg.optimization.micro_batch_size,
                    data_args=cfg.data,
                    data_collator=collator,
                    py_logger=logger,
            )
        else:
            raise NotImplementedError(f"Source {cfg.data.source} not implemented")

    # saving structure assumes deepspeed strategy
    assert isinstance(trainer.strategy, pl.strategies.DeepSpeedStrategy)

    logger.info("*********** data module set up ***********\n\n")
    data_module.setup()



    if not getattr(cfg, "validate_only", False):
        logger.info("*********** start training ***********\n\n")

        trainer.fit(
            model=model_module, datamodule=data_module, ckpt_path=cfg.model.ckpt_path
        )

    else:
        logger.info("*********** start validation***********\n\n")
        trainer.validate(
            model=model_module, datamodule=data_module, ckpt_path=cfg.model.ckpt_path
        )

    if th.distributed.get_rank() == 0:
        # convenient for packaging
        logger.info("Saving huggingface autotokenizer on first rank")
        tokenizer.save_pretrained("autotokenizer")
        logger.info("Finished saving tokenizer")
    th.distributed.barrier()
    logger.info(f"Saving final model weights to {save_dir_path}")
    trainer.save_checkpoint(save_dir_path, weights_only=True)
    logger.info("Finished saving final model")
    # assumes deepspeed strategy, need to prevent hang by saving on all ranks
    # Barrier avoids checkpoint corruption if node 0 exits earlier than other
    # nodes, which can trigger worker node termination
    th.distributed.barrier()

if __name__ == "__main__":
    # Set --replace_sampler_ddp False using SequentialSampler
    # due to index file dependency
    main()
