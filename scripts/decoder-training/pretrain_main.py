# Preliminaries
import hydra
import logging
import os, sys, json
import numpy as np

import torch
import pytorch_lightning as pl
import transformers
from mstar.models.gpt2_model import GPT2Config
from data.data_module import PlDataModule

# add paths for loading modules in script and mstar directory
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(curr_path)
sys.path.append(os.path.dirname(curr_path))
sys.path.append(os.path.dirname(os.path.dirname(curr_path)))

from mstar.utils.lightning import (
    AWSBatchEnvironment,
    KubeFlowEnvironment,
    AWSBatchProgressBar,
    MStarEKSLogger
)


@hydra.main(version_base=None, config_path="config", config_name="base.yaml")
def main(cfg):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Set env variable for using tokenizer in multiple process dataloader
    if cfg.data.num_workers > 0:
        os.environ['TOKENIZERS_PARALLELISM'] = "true"


    logger.info(f"Received ckpt path keyword {cfg.model.ckpt_path}")
    if cfg.model.ckpt_path:
        if cfg.model.ckpt_path=="auto":
            import utils
            logger.info(f"Searching for checkpoint with keyword auto")
            #will either return most recent checkpoint or None if no ckpt present
            cfg.model.ckpt_path = utils.auto_restart.latest_ckpt_wrapper_from_cfg(cfg)

            if cfg.model.ckpt_path is None:
                logger.info(f"Found no checkpoint using keyword auto, starting from scratch")

            else:
                logger.info(f"Resuming from model checkpoint {cfg.model.ckpt_path}")
                cfg.data.data_state_path = os.path.join(cfg.model.ckpt_path,'checkpoint/zero_pp_rank_0_mp_rank_00_model_states.pt')
                logger.info(f"Resuming from data state path {cfg.data.data_state_path}")

        else:
            logger.info(f"Resuming from checkpoint {cfg.model.ckpt_path}")
    else:
        logger.info("No checkpoint passed, starting training from scratch")
       
    # Set seed before initializing model.
    pl.utilities.seed.seed_everything(cfg.optimizer.seed)

    # Load tokenizer and model
    if cfg.data.tokenizer == 'gpt2':
        tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
        if cfg.data.add_special_tokens:
            tokenizer.add_special_tokens({'sep_token': '[SEP]'})
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    elif cfg.data.tokenizer == 'cw':
        sys.path.append('cw_tokenizer')
        from cw_tokenizer.tokenization_vectorbart import VectorBartTokenizer
        tokenizer = VectorBartTokenizer.from_pretrained(cfg.data.tokenizer_path)
    elif cfg.data.tokenizer == 'mstar':
        from mstar.tokenizers import sentencepiece
        tokenizer = sentencepiece.NFDSentencepieceTokenizer(cfg.data.tokenizer_path)
    elif cfg.data.tokenizer == 'mstar-t5':
        from mstar import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer_path)
    else:
        raise NotImplementedError
    logger.info(f'loaded {cfg.data.tokenizer} tokenizer for training\n')

    config_kwargs = {
        "cache_dir": cfg.model.cache_dir,
        "max_length": cfg.data.max_seq_length,
        "length_penalty": cfg.data.length_penalty
    }

    with open(cfg.model.arch_path) as infile:
        model_config = json.load(infile)
        logger.info(f"Loaded config from {cfg.model.arch_path}:\n{model_config}")
    model_config.update(
        {
            "vocab_size": len(tokenizer),
            "fused_scaled_masked_softmax": cfg.model.fused_scaled_masked_softmax,
            "xformers_flash_attention": cfg.model.xformers_flash_attention,
            "fused_gelu": cfg.model.fused_gelu,
            "gradient_checkpointing": cfg.model.gradient_checkpointing,
            "precision": cfg.trainer.precision,
            "positional_embedding": cfg.model.positional_embedding
        }
    )

    # update std for weight initialization
    if cfg.model.scale_init_std:
        new_std = np.sqrt(0.4/model_config['n_embd'])
        model_config.update({"initializer_range": new_std})
        logger.info(f"Updated standard deviation for random weight initialization to {new_std}\n")

    if cfg.data.tokenizer == 'mstar':
        model_config.update(
            {
                "bos_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id
            }
        )


    config = GPT2Config(**{**model_config, **config_kwargs})
    logger.info(f"Loaded GPT model config: {config}\n")

    # Infer number of nodes
    if "AWS_BATCH_JOB_NUM_NODES" in os.environ:
        num_nodes = cfg.trainer.num_nodes
        batch_num_nodes = int(os.environ["AWS_BATCH_JOB_NUM_NODES"])
        if num_nodes != batch_num_nodes:
            logging.warning(
                f"--trainer.num_nodes={num_nodes} != "
                f"$AWS_BATCH_JOB_NUM_NODES={batch_num_nodes}. "
                f"Setting --trainer.num_nodes={batch_num_nodes}!"
            )
            cfg.trainer.num_nodes = batch_num_nodes

    if "KUBERNETES_SERVICE_HOST" in os.environ:
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
        py_logger=logger,
    )

    if cfg.model_module_type=='base':
        from model.model_module import PlModel
    elif cfg.model_module_type=='tok_exp':
        from model.model_tok_exp import PlModel
    else:
        raise NotImplementedError(f'input model_module_type={cfg.model_module_type} is not implemented\n \
                                    use one of the following: base, tok_exp')
    model_module = PlModel(
        config=config,
        tokenizer=tokenizer,
        py_logger=logger,
        optimizer_cfg=cfg.optimizer,
        model_args=cfg.model,
        save_every_n_steps=cfg.callback.save_every_n_steps,
    )

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            every_n_train_steps=cfg.callback.save_every_n_steps,
            save_top_k=cfg.callback.save_top_k,
            monitor="validation_loss",
        ),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ]
    plugins = []
    if "AWS_BATCH_JOB_ID" in os.environ:
        callbacks.append(AWSBatchProgressBar(refresh_rate=25))
        plugins.append(AWSBatchEnvironment(master_port=1337))
    elif "KUBERNETES_SERVICE_HOST" in os.environ:
        plugins.append(KubeFlowEnvironment(master_port=23456))
        if not cfg.run_name:
            run_name = '{}-input-{}'.format(cfg.model.model_type, cfg.data.max_seq_length)
        else:
            run_name = cfg.run_name
        mstar_logger = MStarEKSLogger(experiment_name=cfg.experiment_name,
                                      run_name=run_name, tags={"mode": "Training"})
    if cfg.trainer.gpus == -1:
        cfg.trainer.gpus = torch.cuda.device_count()

    # use trainer.num_sanity_val_steps=0 with zero-2d, otherwise the following error will happen
    # https://github.com/microsoft/DeepSpeed/issues/1938
    '''
    if cfg.trainer.accelerator == "deepspeed":
        print('using deepspeed ...')
        plugins.append(
            pl.plugins.training_type.DeepSpeedPlugin(
                config=cfg.deepspeed_config,
                remote_device=None  # Initialize directly on GPUs instead of CPU (ZeRO-3)
            )
        )
    '''
    strategy = pl.strategies.DeepSpeedStrategy(
                config=cfg.deepspeed_config,
                remote_device=None  # Initialize directly on GPUs instead of CPU (ZeRO-3)
            )
    trainer = pl.Trainer(**cfg.trainer,
                         callbacks=callbacks,
                         plugins=plugins,
                         strategy=strategy,
                         logger=mstar_logger)

    logger.info(f"*********** data module set up ***********\n\n")
    data_module.setup()
    logger.info(f"*********** start training ***********\n\n")
    trainer.fit(model=model_module, datamodule=data_module, ckpt_path=cfg.model.ckpt_path)

    '''
    # for saving the last checkpoint in case its not done by default. also we save only the weights
    # because with deepspeed zero-3 or zero-2d, it saves optimizer parameters along with model.
    if trainer.is_global_zero or cfg.trainer.accelerator == "deepspeed":
        # if trainer.is_global_zero or deepspeed_config:
        # save_checkpoint on all rank with deepspeed to avoid hang
        # https://github.com/microsoft/DeepSpeed/issues/797
        save_path = os.path.join(trainer.default_root_dir, "last.ckpt")
        logger.info(f"Saving model to {save_path}")
        trainer.save_checkpoint(save_path, weights_only=True)
        logger.info("Finished saving")
        if cfg.trainer.accelerator == "deepspeed":
            # Avoid checkpoint corruption if node 0 exits earlier than other
            # nodes triggering termination of other nodes
            torch.distributed.barrier()
    '''

    save_path = os.path.join(trainer.default_root_dir, "last.ckpt")
    logger.info(f"Saving model to {save_path}")
    trainer.save_checkpoint(save_path, weights_only=True)
    logger.info("Finished saving")
    torch.distributed.barrier()


if __name__ == "__main__":
    """process the data per batch; partition data indices files. 
        Set --replace_sampler_ddp False using SequentialSampler
    """
    main()


