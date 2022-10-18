"""
DataModule for working with 1 arrow file for train/val
"""
import pathlib
import glob
import pytorch_lightning as pl
import pyarrow as pa
import torch as th
import numpy as np
from torch.utils.data import DataLoader


class IndexDataset(th.utils.data.Dataset):
    """
    Wrapper class to hold arrow file dataset indices
    """

    def __init__(self, dataset_indices):
        self.dataset_indices = dataset_indices

    def __getitem__(self, index):
        return self.dataset_indices[index]

    def __len__(self):
        return len(self.dataset_indices)


class PlDataModule(pl.LightningDataModule):
    """
    Datamodule to perform pretraining
    based on 1 train arrow file, 1 val arrow file
    Assumes that pre-processed indices exist
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
        if stage is not None:
            self.py_logger.warning(f"Setting data module stage {stage} has no effect")

        self.py_logger.info("Create memory map\n")
        mmap = pa.memory_map(self.training_dataset_path)
        self.py_logger.info("MMAP Read ALL")
        self._train_dataset = pa.ipc.open_stream(mmap).read_all()

        # Check table has single chunk
        # https://issues.apache.org/jira/browse/ARROW-11989
        assert len(self._train_dataset["text"].chunks) == 1
        eval_mmap = pa.memory_map(self.validation_dataset_path)
        self._eval_dataset = pa.ipc.open_stream(eval_mmap).read_all()
        assert len(self._eval_dataset["text"].chunks) == 1

        # Read the pre-generated randomized index files
        # Each GPU reads one train, one val index file
        self.train_files = sorted(
            glob.glob(
                str(pathlib.Path(self.data_args.train_data_index_path) / "*.mmap"),
                recursive=True,
            )
        )
        self.valid_files = sorted(
            glob.glob(
                str(pathlib.Path(self.data_args.valid_data_index_path) / "*.mmap"),
                recursive=True,
            )
        )

    def on_load_checkpoint(self, checkpoint):
        self.py_logger.warning(
            "Reloading method assumes that the previous batchsize was constant throughout training"
        )
        self.resume_index = (
            checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["total"][
                "completed"
            ]
            * self.micro_batch_size
        )

    def train_dataloader(self):
        """This will be run every epoch."""
        process_global_rank = th.distributed.get_rank()
        # read the index in 'r+' mode, as we need to shuffle it later
        dataset_indices = np.memmap(
            self.train_files[process_global_rank], dtype="uint32", mode="r+"
        )

        # shuffle the indices for every epoch other than 0.
        # the loaded indices are already shuffled
        if self.trainer.current_epoch > 0:
            seed = self.seed + self.trainer.current_epoch
            rng = np.random.default_rng(seed)
            rng.shuffle(dataset_indices)

        if self.resume_index is not None:
            dataset_indices = dataset_indices[self.resume_index :]
            self.resume_index = None  # reset to avoid next-epoch issues

        train_index_dataset = IndexDataset(dataset_indices)

        def train_pl_collate_fn(indices):
            # if inputs are too long tokenizer needs to truncate
            inputs = self._train_dataset.take(indices)["text"].to_pylist()
            return self._data_collator(inputs)

        loader = DataLoader(
            train_index_dataset,
            batch_size=self.micro_batch_size,
            collate_fn=train_pl_collate_fn,
            num_workers=self.data_args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.py_logger.info("Finished loading training data")
        return loader

    def val_dataloader(self):

        process_global_rank = th.distributed.get_rank()

        dataset_indices = np.memmap(
            self.valid_files[process_global_rank], dtype="uint32", mode="r"
        )

        valid_index_dataset = IndexDataset(dataset_indices)

        def val_pl_collate_fn(indices):
            inputs = self._eval_dataset.take(indices)["text"].to_pylist()
            return self._data_collator(inputs)

        loader = DataLoader(
            valid_index_dataset,
            batch_size=self.micro_batch_size,
            collate_fn=val_pl_collate_fn,
            num_workers=self.data_args.num_workers,
            drop_last=True,
        )
        self.py_logger.info("Finished loading validation data")
        return loader
