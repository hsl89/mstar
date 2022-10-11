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
from mstar.optimizers import FusedAdam
from mstar.utils.lightning import KubeFlowEnvironment, MStarEKSLogger
import torch as th
import transformers
import pytorch_lightning as pl


# local imports
import models
from data.datamodule import PlDataModule
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

    logger.info("Optimizer args")
    logger.info(cfg.optimizer)
    logger.info("Data args")
    logger.info(cfg.data)
    logger.info("Model args")
    logger.info(cfg.model)
    logger.info("Trainer args")
    logger.info(cfg.trainer)
    logger.info("Callback args")
    logger.info(cfg.callback)
    logger.info("Deepspeed args")
    logger.info(cfg.deepspeed)

    # If dataloader is already multiprocessing, skip this
    if cfg.data.num_workers > 0:
        os.environ["TOKENIZERS_PARALLELISM"] = "False"

    # Set seed before initializing model
    pl.utilities.seed.seed_everything(cfg.optimizer.seed)

    # set up huggingface-style model config
    # uses custom config class for extra args
    hf_model_config = models.configuration_t5.MStarT5Config(**cfg.model)

    tokenizer, collator = collators.solver.get_collator(
        data_args=cfg.data,
        decoder_start_token_id=hf_model_config.decoder_start_token_id,
    )

    # make sure that model has enough embeddings for the tokenizer
    if hf_model_config.vocab_size < len(tokenizer):
        hf_model_config.vocab_size, old_embedding_number = (
            len(tokenizer),
            hf_model_config.vocab_size,
        )
        logger.info(
            f"Updated model vocab size from {old_embedding_number} "
            f"to {hf_model_config.vocab_size} to cover vocab size"
        )

    # make sure embeddings are multiple of 128 for high tensor core efficiency
    if hf_model_config.vocab_size % 128 != 0:
        hf_model_config.vocab_size, old_embedding_number = (
            int(np.ceil(hf_model_config.vocab_size / 128)) * 128,
            hf_model_config.vocab_size,
        )
        logger.info(
            f"Updated model vocab size from {old_embedding_number} "
            f"to {hf_model_config.vocab_size} to make multiple of 128"
        )

    if cfg.model.fused_scaled_masked_softmax:
        # the information assigned below is necessary
        # for the mstar megatron softmax
        assert cfg.trainer.precision in [16, "bf16"]
        softmax_precision = (
            "fp16" if cfg.trainer.precision == 16 else cfg.trainer.precision
        )
        setattr(hf_model_config, "softmax_precision", softmax_precision)

    # TODO:using custom model class, port to mstar
    model = models.t5_model.T5ForConditionalGeneration(config=hf_model_config)

    if cfg.model.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # caching encoder states not compatible with checkpointing
        model.config.use_cache = False

    assert "KUBERNETES_SERVICE_HOST" in os.environ, "Only support EKS cluster"
    num_nodes = cfg.trainer.num_nodes
    kubeflow_num_nodes = int(os.environ["NUM_NODES"])
    if num_nodes != kubeflow_num_nodes:
        logging.warning(
            f"--trainer.num_nodes={num_nodes} != "
            f"$NUM_NODES={kubeflow_num_nodes}. "
            f"Setting --trainer.num_nodes={kubeflow_num_nodes}!"
        )
        cfg.trainer.num_nodes = kubeflow_num_nodes

    data_module = PlDataModule(
        tokenizer=tokenizer,
        training_dataset_path=cfg.data.training_dataset,
        validation_dataset_path=cfg.data.validation_dataset,
        seed=cfg.optimizer.seed,
        batch_size=cfg.optimizer.batch_size,
        data_args=cfg.data,
        data_collator=collator,
        py_logger=logger,
    )

    model_module = models.modelmodule.PlModel(
        config=hf_model_config,
        model=model,
        py_logger=logger,
        optimizer_cfg=cfg.optimizer,
        scheduler_mult_factor=cfg.optimizer.scheduler_mult_factor,  # used to resume from a checkpoint with an adjusted scheduler, will reload scheduler otherwise
    )

    plugins = []

    assert "KUBERNETES_SERVICE_HOST" in os.environ, "Only set up for mstar cluster"
    plugins.append(KubeFlowEnvironment(master_port=23456))

    # assumes EKS cluster usage
    mstar_logger = MStarEKSLogger(
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
        tags={"mode": "Training"},
    )

    save_dir_path = utils.logging.get_save_dir(cfg)  # , mstar_logger)

    if cfg.trainer.gpus == -1:
        cfg.trainer.gpus = th.cuda.device_count()

    strategy = pl.strategies.DeepSpeedStrategy(
        **cfg.deepspeed,
        remote_device=None,  # Initialize directly on GPUs instead of CPU (ZeRO-3)
    )

    # we directly read mmap files for each gpu
    # don't let PTL replace sampler since we
    # already handle distributed sampling
    assert cfg.trainer.replace_sampler_ddp is False

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            every_n_train_steps=cfg.callback.save_every_n_train_steps,
            save_top_k=cfg.callback.save_top_k,
            dirpath=save_dir_path,
            monitor="validation_loss",
            mode="min",
            filename="{epoch}-{step}-{validation_loss:.4f}_{training_loss_step:.4f}",
            save_last=True,
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        **cfg.trainer,
        accelerator="gpu",
        callbacks=callbacks,
        plugins=plugins,
        strategy=strategy,
        logger=mstar_logger,
    )

    # saving structure assumes deepspeed strategy
    assert isinstance(trainer.strategy, pl.strategies.DeepSpeedStrategy)

    logger.info("*********** data module set up ***********\n\n")
    data_module.setup()
    logger.info("*********** start training ***********\n\n")

    if cfg.model.ckpt_path:
        logger.info(f"Resuming from checkpoint to {cfg.model.ckpt_path}")

    trainer.fit(
        model=model_module, datamodule=data_module, ckpt_path=cfg.model.ckpt_path
    )

    # assumes deepspeed strategy, need to prevent hang by saving on all ranks
    logger.info(f"Saving model to {save_dir_path}")
    trainer.save_checkpoint(save_dir_path, weights_only=True)
    logger.info("Finished saving model weights")
    # Barrier avoids checkpoint corruption if node 0 exits earlier than other
    # nodes triggering termination of other nodes
    th.distributed.barrier()

    # TODO: block below may not work for larger models
    # but is convenient for non-sharded models
    if th.distributed.get_rank() == 0:
        logger.info("Saving huggingface autotokenizer on first rank")
        tokenizer.save_pretrained(os.path.join(save_dir_path, "tokenizer"))
        logger.info("Finished saving tokenizer")
        try:
            logger.info("Saving huggingface autotokenizer on first rank")
            model.save_pretrained(os.path.join(save_dir_path, "hf_version"))
            logger.info("Finished Saving huggingface automodel")
        except:
            logger.info("Failed to save huggingface automodel")

    th.distributed.barrier()


if __name__ == "__main__":
    # Set --replace_sampler_ddp False using SequentialSampler
    # due to index file dependency
    main()
