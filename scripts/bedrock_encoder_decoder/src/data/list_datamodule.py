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
        training_datasets,
        validation_datasets,
        seed,
        micro_batch_size,
        data_args,
        data_collator,
        py_logger,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.training_dataset_paths = training_datasets
        self.validation_dataset_paths = validation_datasets
        self.seed = seed
        self.micro_batch_size = micro_batch_size
        self.training_data = None
        self.data_args = data_args
        self.py_logger = py_logger

        if hasattr(data_args, "resume_index"):
            self.py_logger.info("Starting from resume dataset index")

            self.resume_index = data_args.resume_index * self.micro_batch_size
        else:
            self.resume_index = None

        # self.save_hyperparameters()

        self._data_collator = data_collator

    def on_load_checkpoint(self, checkpoint):
        self.py_logger.warning(
            "Reloading method assumes that the previous batchsize was constant throughout training"
        )
        # how far to increment the dataloader
        self.resume_index = (
            checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["total"][
                "completed"
            ]
            * self.micro_batch_size
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

        self.py_logger.info(
            f"Loading training datasets from paths {self.training_dataset_paths}"
        )

        train_datasets = [
            datasets.arrow_dataset.Dataset.from_file(x)
            .shard(num_shards=world_size, index=process_global_rank)
            .shuffle(seed=self.trainer.current_epoch + self.seed)
            for x in self.training_dataset_paths
        ]

        if 'sampling_prob' in self.data_args:
            self.py_logger.info(
                f"Sampling training datasets with probabilities {self.data_args.sampling_prob}"
            )
            train_dataset = datasets.interleave_datasets(
                train_datasets, probabilities=self.data_args.sampling_prob,
                seed=self.trainer.current_epoch + self.seed, stopping_strategy="first_exhausted")
        else:
            train_dataset = datasets.concatenate_datasets(train_datasets).shuffle()

        if self.resume_index is not None:
            # in case we are past the first epoch, need modulo
            self.resume_index = self.resume_index % len(train_dataset)
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

        val_datasets = [
            datasets.arrow_dataset.Dataset.from_file(x)
            .shard(num_shards=world_size, index=process_global_rank)
            .shuffle(seed=self.trainer.current_epoch + self.seed)
            for x in self.validation_dataset_paths
        ]

        # shuffle again in case we don't do a full val run
        val_dataset = datasets.concatenate_datasets(val_datasets).shuffle()

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


