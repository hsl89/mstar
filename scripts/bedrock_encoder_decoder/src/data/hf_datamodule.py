"""
DataModule for working with 1 arrow file for train/val
"""
import pathlib
import glob
import datasets
import pytorch_lightning as pl
import pyarrow as pa
import torch as th
import numpy as np
from torch.utils.data import DataLoader


class HFDataModule(pl.LightningDataModule):
    """
    Datamodule to perform pretraining
    based on 1 train arrow file, 1 val arrow file
    """

    def __init__(
        self,
        tokenizer,
        training_dataset_path,
        validation_dataset_path,
        seed,
        micro_batch_size,
        data_args,
        data_collator,
        py_logger,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.training_dataset_path = training_dataset_path
        self.validation_dataset_path = validation_dataset_path
        self.seed = seed
        self.micro_batch_size = micro_batch_size
        self.training_data = None
        self.data_args = data_args
        self.py_logger = py_logger
        self.resume_index = None

        # self.save_hyperparameters()

        self._data_collator = data_collator

    def setup(self, stage=None):
        """
        Check chunk sizes 
        """
        if stage is not None:
            self.py_logger.warning(f"Setting data module stage {stage} has no effect")

        self.py_logger.info("Create training memory map to check chunksize\n")
        mmap = pa.memory_map(self.training_dataset_path)
        self.py_logger.info("Train MMAP Read ALL")
        train_dataset = pa.ipc.open_stream(mmap).read_all()

        # Check table has single chunk
        # https://issues.apache.org/jira/browse/ARROW-11989
        assert len(train_dataset["text"].chunks) == 1
        self.py_logger.info("Create eval memory map to check chunksize\n")
        eval_mmap = pa.memory_map(self.validation_dataset_path)
        self.py_logger.info("Eval MMAP Read ALL")
        eval_dataset = pa.ipc.open_stream(eval_mmap).read_all()
        assert len(eval_dataset["text"].chunks) == 1

    def on_load_checkpoint(self, checkpoint):
        self.py_logger.warning(
            "Reloading method assumes that the previous batchsize was constant throughout training"
        )
        # how far to increment the dataloader
        self.resume_index = (
            checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["total"][
                "completed"
            ]
            * checkpoint["hyper_parameters"]["optimizer_cfg"].micro_batch_size
        )

    def collate_fn(self, x):
        return self._data_collator([y["text"] for y in x])

    def train_dataloader(self):
        """This will be run every epoch."""

        process_global_rank = th.distributed.get_rank()
        world_size = th.distributed.get_world_size()
        self.py_logger.info(
            f"Loading training shard {process_global_rank} with world size {world_size}"
        )

        train_dataset = (
            datasets.arrow_dataset.Dataset.from_file(self.training_dataset_path)
            .shard(num_shards=world_size, index=process_global_rank)
            .shuffle(seed=self.trainer.current_epoch + self.seed)
        )

        if self.resume_index is not None:
            # increments the dataset indices we have already covered
            dataset_indices = range(self.resume_index, len(train_dataset))
            train_dataset = train_dataset.select(dataset_indices)
            self.resume_index = None  # reset to avoid next-epoch issues

        loader = DataLoader(
            train_dataset,
            batch_size=self.micro_batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.data_args.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=4,
        )
        self.py_logger.info("Finished loading training data")
        return loader

    def val_dataloader(self):

        process_global_rank = th.distributed.get_rank()
        world_size = th.distributed.get_world_size()
        self.py_logger.info(
            f"Loading validation shard {process_global_rank} with world size {world_size}"
        )

        val_dataset = (
            datasets.arrow_dataset.Dataset.from_file(self.validation_dataset_path)
            .shard(num_shards=world_size, index=process_global_rank)
            .shuffle(seed=self.trainer.current_epoch + self.seed)
        )

        loader = DataLoader(
            val_dataset,
            batch_size=self.micro_batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.data_args.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

        self.py_logger.info("Finished loading validation data")
        return loader
