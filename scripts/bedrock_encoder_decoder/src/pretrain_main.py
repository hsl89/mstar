"""
Main script to launch pretraining
"""

# Preliminaries
import hydra
import yaml
import logging
import os
import sys
import json
from typing import Optional
from dataclasses import dataclass, field, asdict
import numpy as np
import mstar.models.t5
import mstar.AutoTokenizer
from mstar.optimizers import FusedAdam
from mstar.utils.lightning import KubeFlowEnvironment, MStarEKSLogger
import torch as th
import transformers
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar
import deepspeed

# local imports
import models
import my_tokenizer
import data
import collators
import utils


@hydra.main(version_base=None, config_path="config", config_name="base.yaml")
def main(cfg):
    """
    Launch pretraining
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    logger.info("Configuration args")
    logger.info(cfg)

    # If dataloader is already multiprocessing, skip this
    if cfg.data.num_workers > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "False"

    assert "KUBERNETES_SERVICE_HOST" in os.environ, "Only support EKS cluster"
    kubeflow_num_nodes = int(os.environ["NUM_NODES"])

    computed_total_batch_size = (
        cfg.trainer.num_nodes * cfg.optimizer.micro_batch_size * th.cuda.device_count()
    )
    logging.info(
        f"Training with {cfg.trainer.num_nodes} nodes "
        f"micro-batch size {cfg.optimizer.micro_batch_size} "
        f"total batch size {computed_total_batch_size}"
        f"and {th.cuda.device_count()} devices per-node"
    )
    
    # Set seed before initializing model
    pl.utilities.seed.seed_everything(cfg.optimizer.seed)

    if hasattr(cfg.data, "autotokenizer_path"):
        logger.info(
            f"Loading autotokenizer from specified tokenzier path {cfg.data.autotokenizer_path}"
        )
        tokenizer = mstar.AutoTokenizer.from_pretrained(cfg.data.autotokenizer_path)
    else:
        tokenizer = my_tokenizer.solver.get_tokenizer(data_args=cfg.data)

    # set up huggingface-style model config
    # uses custom config class for extra args
    hf_model_config = mstar.models.t5.MStarT5Config(**cfg.model)

    collator = collators.solver.get_collator(
        data_args=cfg.data,
        tokenizer=tokenizer,
        decoder_start_token_id=hf_model_config.decoder_start_token_id,
    )

    # make sure that model has enough embeddings for the tokenizer
    assert hf_model_config.vocab_size >= len(tokenizer), f"Model vocab size {hf_model_config.vocab_size} too small for tokenizer vocab size {len(tokenizer)}"

    if cfg.model.fused_scaled_masked_softmax:
        #make sure trainer/softmax precision match
        if cfg.trainer.precision==16:
            #16=fp16 for pytorch lightning
            assert hf_model_config.softmax_precision=="fp16", f"Trainer precision {cfg.trainer.precision} should match softmax precision {hf_model_config.softmax_precision}"
        elif cfg.trainer.precision=="bf16":
            assert hf_model_config.softmax_precision=="bf16", f"Trainer precision {cfg.trainer.precision} should match softmax precision {hf_model_config.softmax_precision}"
        else:
            raise ValueError(f"Trainer precision {cfg.trainer.precision} does not match any softmax precision")


    plugins = []
    plugins.append(KubeFlowEnvironment(master_port=23456))

    # assumes EKS cluster usage
    mstar_logger = MStarEKSLogger(
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
        tags={"mode": "Training"},
        s3_upload=False,  #slows down large model training
    )

    save_dir_path = utils.logging.get_save_dir(cfg)

    if cfg.trainer.gpus == -1:
        cfg.trainer.gpus = th.cuda.device_count()

    # strategy determines distributed training
    # required to use deepspeed config json
    # used for optimized Zero-2D internal version
    strategy = pl.strategies.DeepSpeedStrategy(
        config=cfg.deepspeed_path,
        remote_device=None,  # Initialize directly on GPUs instead of CPU (ZeRO-3)
    )

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            every_n_train_steps=cfg.callback.save_every_n_train_steps,
            save_top_k=cfg.callback.save_top_k,
            dirpath=save_dir_path,
            monitor="validation_loss",
            mode="min",
            filename="{epoch}-{step}-{validation_loss:.2f}",
            save_last=True,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        #automatically stop job on nan
        pl.callbacks.EarlyStopping(
            monitor="training_loss_step",  # monitor this logged value
            patience=10000000,  # don't actually stop on train 
            strict=True,  # monitored value must exist
            check_finite=True,  # forces monitored value to be finite
        ),
        TQDMProgressBar(refresh_rate=cfg.trainer.log_every_n_steps),
    ]

    trainer = pl.Trainer(
        **cfg.trainer,
        accelerator="gpu",
        callbacks=callbacks,
        plugins=plugins,
        strategy=strategy,
        logger=mstar_logger,
    )
    
    assert len(list(filter(None,[cfg.model.state_dict_path,cfg.model.ckpt_path])))<=1, "Resume from either cfg.model.state_dict_path or cfg.model.ckpt_path not both"

    model_init_fn = lambda : models.utils.load_model(
        trainer=trainer,
        precision=cfg.trainer.precision, 
        model_config=hf_model_config,
        state_dict_path=cfg.model.state_dict_path,
    )

    model_module = models.modelmodule.PlModel(
        config=hf_model_config,
        full_experiment_config=cfg,  # pass full cfg over for easier logging
        model_init_fn=model_init_fn,
        py_logger=logger,
        optimizer_cfg=cfg.optimizer,
        scheduler_mult_factor=cfg.optimizer.scheduler_mult_factor,  # used to resume from a checkpoint with an adjusted scheduler, will reload scheduler otherwise
    )

    data_module = data.solver.get_datamodule(tokenizer, cfg, collator, logger)


    # saving structure assumes deepspeed strategy
    assert isinstance(trainer.strategy, pl.strategies.DeepSpeedStrategy)

    logger.info("*********** data module set up ***********\n\n")
    data_module.setup()


    if cfg.model.ckpt_path:
        if cfg.model.ckpt_path=="auto":
            import fake_auto_restart
            cfg.model.ckpt_path = fake_auto_restart.latest_ckpt_wrapper_from_cfg(cfg)

        logger.info(f"Resuming from checkpoint to {cfg.model.ckpt_path}")

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
