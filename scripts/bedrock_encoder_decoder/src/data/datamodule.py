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

# don't rely on cached index files 
datasets.disable_caching()

def check_arrow_for_single_chunk(filepath:str, key="text"):
    """
    Checks a .arrow file, specified by filepath, to see if it's 1-chunk
    More than 1 chunk reduces .take() performance
    """
    mmap = pa.memory_map(filepath)
    stream = pa.ipc.open_stream(mmap).read_all()

    # Check table has single chunk
    # https://issues.apache.org/jira/browse/ARROW-11989
    assert len(stream[key].chunks) == 1, f"File {filepath} is not a single-chunk arrow file"


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

        self.py_logger.info("Ensuring dataset files all have one chunk")
        # check train and val files for single chunk
        for filepath in [*self.training_dataset_paths,*self.validation_dataset_paths]:
            check_arrow_for_single_chunk(filepath)

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

            #possible to cause a hang if sampling prob gets too low
            #TODO @colehawk more investigation
            EPSILON=0.01
            if min(self.data_args.sampling_prob)<EPSILON:       
           
                new_train_datasets = []
                new_sampling_probs = []
                new_filepaths = []
                
                for prob, dataset, path in zip(self.data_args.sampling_prob, train_datasets, self.training_dataset_paths):
                    if prob>EPSILON:
                        #only keep datasets with enough entries
                        new_train_datasets.append(dataset)
                        new_sampling_probs.append(prob)
                        new_filepaths.append(path)

                #renormalize
                normalization_constant = sum(new_sampling_probs)
                new_sampling_probs=[x/normalization_constant for x in new_sampling_probs]
                train_datasets = new_train_datasets
                sampling_probs = new_sampling_probs

                self.py_logger.warning(
                    f"Ratios were too low for some datasets erasing all datasets with prob < {EPSILON}"
                )
                self.py_logger.warning(
                    f"Using probs {sampling_probs} and files {new_filepaths}"
                )
                #raise ValueError
            else:
                sampling_probs = self.data_args.sampling_prob
            #remove probs that are too low

            train_dataset = datasets.interleave_datasets(
                train_datasets, probabilities=sampling_probs,
                seed=self.trainer.current_epoch + self.seed, stopping_strategy="first_exhausted")
        else:
            self.py_logger.warning(
                "Concatenating arrow datasets can slow down .take() " 
                "Make sure to use pre-fetch+multiprocessing."
            )
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


