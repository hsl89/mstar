"""
Wrapper to handle loading of the pytorch lightning datamodule
"""
from . import p3_datamodule, mixed_datamodule, hf_datamodule, list_datamodule


def get_datamodule(tokenizer, cfg, collator, logger):
    """Wrapper function to load datamodule
    tokenizer: a huggingface pretrained tokenizer
    cfg: a hydra config (nested)
    collator: a HF seq-to-seq collator
    logger: the mstar logger for EKS usage
    """

    if cfg.data.source == "mtl":
        # tokenizer = hf.AutoTokenizer.from_pretrained(TOKENIZER)
        data_module = p3_datamodule.T50Data(
            tokenizer, p3_batch=cfg.optimizer.micro_batch_size, pile_batch=0
        )
        # For quick validation
        assert cfg.trainer.replace_sampler_ddp

    elif cfg.data.source == "mix_from_lists":
        assert cfg.trainer.replace_sampler_ddp is False
        data_module = list_datamodule.HFDataModule(
            tokenizer=tokenizer,
            training_datasets=cfg.data.training_datasets,
            validation_datasets=cfg.data.validation_datasets,
            seed=cfg.optimizer.seed,
            micro_batch_size=cfg.optimizer.micro_batch_size,
            data_args=cfg.data,
            data_collator=collator,
            py_logger=logger,
        )
    elif cfg.data.source == "unlabeled_mixed":
        # only used for reddit right now
        assert cfg.trainer.replace_sampler_ddp is False
        data_module = mixed_datamodule.HFDataModule(
            tokenizer=tokenizer,
            training_dataset_path1=cfg.data.training_dataset1,
            training_dataset_path2=cfg.data.training_dataset2,
            validation_dataset_path1=cfg.data.validation_dataset1,
            seed=cfg.optimizer.seed,
            micro_batch_size=cfg.optimizer.micro_batch_size,
            data_args=cfg.data,
            data_collator=collator,
            py_logger=logger,
        )

    elif cfg.data.source == "unlabeled":
        # necessary for our pile loading setup
        assert not cfg.trainer.replace_sampler_ddp
        data_module = hf_datamodule.HFDataModule(
            tokenizer=tokenizer,
            training_dataset_path=cfg.data.training_dataset,
            validation_dataset_path=cfg.data.validation_dataset,
            seed=cfg.optimizer.seed,
            micro_batch_size=cfg.optimizer.micro_batch_size,
            data_args=cfg.data,
            data_collator=collator,
            py_logger=logger,
        )
    else:
        raise NotImplementedError

    return data_module
