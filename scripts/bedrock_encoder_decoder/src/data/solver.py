"""
Wrapper to handle loading of the pytorch lightning datamodule
"""
import data.datamodule

def get_datamodule(tokenizer, cfg, collator, logger):
    """Wrapper function to load datamodule
    tokenizer: a huggingface pretrained tokenizer
    cfg: a hydra config (nested)
    collator: a HF seq-to-seq collator
    logger: the mstar logger for EKS usage
    """

    assert cfg.trainer.replace_sampler_ddp is False
    data_module = data.datamodule.HFDataModule(
        tokenizer=tokenizer,
        training_datasets=cfg.data.training_datasets,
        validation_datasets=cfg.data.validation_datasets,
        seed=cfg.optimizer.seed,
        micro_batch_size=cfg.optimizer.micro_batch_size,
        data_args=cfg.data,
        data_collator=collator,
        py_logger=logger,
    )

    return data_module
