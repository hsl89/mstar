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
import models
import mstar.AutoTokenizer
import mstar.AutoModel
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

    # TODO:using custom model class, port to mstarmodel factory
    if getattr(cfg.model, "load_method", None) == "automodel":
        logger.info(f"Loading pretrained model from {cfg.model.automodel_path}")
        logger.info(f"Make sure to also load pretrained tokenizer")
        model_init_fn = lambda: mstar.AutoModel.from_pretrained(
            cfg.model.automodel_path
        )

    elif getattr(cfg.model, "load_method", None) == "state_dict":
        # first requires the user to pre-package
        # state dict from deepspeed zero ckpt
        logger.info(f"Loading from state dict")

        def tmp_init():

            state_dict = th.load(cfg.model.state_dict_path, map_location="cpu")
            try:
                state_dict.pop(
                    "encoder.block.0.layer.0.SelfAttention.alibi_positional_bias"
                )
                state_dict.pop(
                    "decoder.block.0.layer.0.SelfAttention.alibi_positional_bias"
                )
            except:
                pass
            # Need to make sure vocab sizes match, then load from checkpoint

            # get model vocab size from embedding table
            hf_model_config.vocab_size = state_dict["shared.weight"].shape[0]
            hf_model_config.gradient_checkpointing = True
            hf_model_config.use_cache = False
            # load model on CPU for more memory
            model = mstar.models.t5.MStarT5ForConditionalGeneration(
                config=hf_model_config
            ).cpu()
            model.load_state_dict(state_dict, strict=False)
            model.gradient_checkpointing_enable()

            return model

        model_init_fn = tmp_init

    elif getattr(cfg.model, "load_method", None) == "safe_state_dict":
        logger.info(
            f"loading the model parameters from the state dict file {cfg.model.state_dict_path!r}"
        )

        def model_init_fn(trainer):
            VOCAB_SIZE = 34176
            unwrapped_state_dict = None
            if trainer.is_global_zero:
                unwrapped_state_dict = th.load(
                    cfg.model.state_dict_path, map_location="cpu"
                )
                assert unwrapped_state_dict["shared.weight"].shape[0] == VOCAB_SIZE

            def load(module: th.nn.Module, prefix=""):
                nonlocal unwrapped_state_dict
                missing_keys = []
                unexpected_keys = []
                error_msgs = []
                # copy state_dict so _load_from_state_dict can modify it
                metadata = getattr(unwrapped_state_dict, "_metadata", None)
                state_dict = None
                if trainer.is_global_zero:
                    state_dict = unwrapped_state_dict.copy()

                    if metadata is not None:
                        state_dict._metadata = metadata

                local_metadata = (
                    {} if metadata is None else metadata.get(prefix[:-1], {})
                )
                # because zero3 puts placeholders in model params, this context
                # manager gathers (unpartitions) the params of the current layer, then loads from
                # the state dict and then re-partitions them again
                with deepspeed.zero.GatheredParameters(
                    list(module.parameters(recurse=False)), modifier_rank=0
                ):
                    if trainer.is_global_zero:
                        module._old_load_from_state_dict(
                            state_dict=state_dict,
                            prefix=prefix,
                            local_metadata=local_metadata,
                            strict=True,
                            missing_keys=missing_keys,
                            unexpected_keys=unexpected_keys,
                            error_msgs=error_msgs,
                        )

                for name, child in module._modules.items():
                    if child is not None:
                        load(child, prefix + name + ".")

            # get model vocab size from embedding table
            hf_model_config.vocab_size = VOCAB_SIZE
            hf_model_config.gradient_checkpointing = True
            hf_model_config.use_cache = False

            # quick fix for bedrock, bf16 is not actually a requirement
            init_dtype = th.bfloat16
            assert hf_model_config.softmax_precision == "bf16"
            assert trainer.precision == "bf16"
            init_dtype = th.bfloat16
            # load model on CPU for more memory
            context = deepspeed.zero.Init(
                remote_device=trainer.training_type_plugin.remote_device,
                pin_memory=True,
                config=trainer.training_type_plugin.config,
                dtype=init_dtype,
            )  # dtype)

            with context:
                model = models.t5.MStarT5ForConditionalGeneration(
                    config=hf_model_config
                )

            load(model, prefix="")

            return model

    else:

        def model_init_fn():
            logger.info(f"Random initialization of untrained model")
            model = mstar.models.t5.MStarT5ForConditionalGeneration(
                config=hf_model_config
            )
            model.gradient_checkpointing_enable()
            hf_model_config.use_cache = False
            assert model.is_gradient_checkpointing
            if hasattr(cfg, "decoder_disable_grad_ckpt"):
                model.decoder.gradient_checkpointing_disable()

            return model

    plugins = []
    plugins.append(KubeFlowEnvironment(master_port=23456))

    # assumes EKS cluster usage
    mstar_logger = MStarEKSLogger(
        experiment_name=cfg.experiment_name,
        run_name=cfg.run_name,
        tags={"mode": "Training"},
        s3_upload=False,  #slows down large model training
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
            filename="{epoch}-{step}-{validation_loss:.4f}_{training_loss_step:.4f}",
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
        logger.info("Saving huggingface autotokenizer on first rank")
        tokenizer.save_pretrained("autotokenizer")
        logger.info("Finished saving tokenizer")
    th.distributed.barrier()
    # assumes deepspeed strategy, need to prevent hang by saving on all ranks
    logger.info(f"Saving final model weights to {save_dir_path}")
    trainer.save_checkpoint(save_dir_path, weights_only=True)
    # Barrier avoids checkpoint corruption if node 0 exits earlier than other
    # nodes, which can trigger worker node termination
    logger.info("Finished saving final model")
    th.distributed.barrier()

    # also save automodel
    if getattr(cfg, "save_automodel_from_state_dict", False):
        if th.distributed.get_rank() == 0:
            logger.info("Saving automodel on rank 0")
            to_save = model_init_fn()
            to_save.save_pretrained(os.path.join(save_dir_path, "hf_automodel"))

    # avoid saving corruption if workers
    # exit early and the job crashes
    th.distributed.barrier()


if __name__ == "__main__":
    # Set --replace_sampler_ddp False using SequentialSampler
    # due to index file dependency
    main()
