"""
DataModule for working with 1 arrow file for train/val
"""

import os
import datasets
import pytorch_lightning as pl
import pyarrow as pa
import torch as th
from torch.utils.data import DataLoader
import data.online_packing

# don't rely on cached index files
datasets.disable_caching()

# used to resume dataloader state
DATALOADER_SAVE_KEY = "dataloaders"


def check_arrow_for_single_chunk(filepath: str, key="text"):
    """
    Checks a .arrow file, specified by filepath, to see if it's 1-chunk
    More than 1 chunk reduces .take() performance
    """
    mmap = pa.memory_map(filepath)
    stream = pa.ipc.open_stream(mmap).read_all()

    # Check table has single chunk
    # https://issues.apache.org/jira/browse/ARROW-11989
    assert (
        len(stream[key].chunks) == 1
    ), f"File {filepath} is not a single-chunk arrow file"


class HFDataModule(pl.LightningDataModule):
    """
    Datamodule to perform pretraining
    based on 1 train arrow file, 1 val arrow file
    """

    def __init__(
        self,
        tokenizer,
        training_datasets: list,
        validation_datasets: list,
        seed: int,
        micro_batch_size: int,
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
            # can be manually passed in, but usually accessed via ckpt load
            self.py_logger.info("Starting from resume dataset index")

            self.resume_index = data_args.resume_index * self.micro_batch_size
        else:
            self.resume_index = None

        self.py_logger.info("Ensuring dataset files all have one chunk")
        # check train and val files for single chunk
        for filepath in [*self.training_dataset_paths, *self.validation_dataset_paths]:
            check_arrow_for_single_chunk(filepath)

        self._data_collator = data_collator

    def setup(self, stage=None):
        pass

    def setup_train_hf_dataset(self):
        """
        Set up train HF datasets that will be consumed by the train dataloader
        """
        process_global_rank = th.distributed.get_rank()
        world_size = th.distributed.get_world_size()
        self.py_logger.info(
            f"Loading training shard {process_global_rank} with world size {world_size}"
        )

        self.py_logger.info(
            f"Loading training datasets from paths {self.training_dataset_paths}"
        )

        train_datasets = [
            datasets.arrow_dataset.Dataset.from_file(x).shard(
                num_shards=world_size, index=process_global_rank
            )
            for x in self.training_dataset_paths
        ]

        if "sampling_prob" in self.data_args:
            self.py_logger.info(
                f"Sampling training datasets with probabilities {self.data_args.sampling_prob}"
            )

            # possible to cause a hang if sampling prob gets too low
            # TODO @colehawk more investigation
            EPSILON = 0.01
            if min(self.data_args.sampling_prob) < EPSILON:

                new_train_datasets = []
                new_sampling_probs = []
                new_filepaths = []

                for prob, dataset, path in zip(
                    self.data_args.sampling_prob,
                    train_datasets,
                    self.training_dataset_paths,
                ):
                    if prob > EPSILON:
                        # only keep datasets with enough entries
                        new_train_datasets.append(dataset)
                        new_sampling_probs.append(prob)
                        new_filepaths.append(path)

                # renormalize
                normalization_constant = sum(new_sampling_probs)
                new_sampling_probs = [
                    x / normalization_constant for x in new_sampling_probs
                ]
                train_datasets = new_train_datasets
                sampling_probs = new_sampling_probs

                self.py_logger.warning(
                    f"Ratios were too low for some datasets erasing all datasets with prob < {EPSILON}"
                )
                self.py_logger.warning(
                    f"Using probs {sampling_probs} and files {new_filepaths}"
                )
                # raise ValueError
            else:
                sampling_probs = self.data_args.sampling_prob
            # remove probs that are too low

            train_dataset = datasets.interleave_datasets(
                train_datasets,
                probabilities=sampling_probs,
                seed=self.trainer.current_epoch + self.seed,
                stopping_strategy="first_exhausted",
            )
        else:
            self.py_logger.warning(
                "Concatenating arrow datasets can slow down .take() "
                "Make sure to use pre-fetch+multiprocessing."
            )
            train_dataset = datasets.concatenate_datasets(train_datasets)

        self.train_hf_dataset = train_dataset.shuffle(seed=self.seed)

    def setup_val_hf_dataset(self):
        """
        Set up val dataset that will be consumed by the val dataloader
        """
        process_global_rank = th.distributed.get_rank()
        world_size = th.distributed.get_world_size()
        self.py_logger.info(
            f"Loading validation shard {process_global_rank} with world size {world_size}"
        )

        val_datasets = [
            datasets.arrow_dataset.Dataset.from_file(x).shard(
                num_shards=world_size, index=process_global_rank
            )
            for x in self.validation_dataset_paths
        ]

        # shuffle in case we limit val batches and data is not pre-shuffled
        self.val_hf_dataset = datasets.concatenate_datasets(val_datasets).shuffle(
            seed=self.seed
        )
  
    
    def state_dict(self):
        """
        Triggered on checkpoint save
        Will not be copied on a per-worker basis
        """
        #get batch idx within one epoch
        return {"batch_idx": self.trainer.fit_loop.epoch_loop.batch_idx, "micro_batch_size": self.micro_batch_size}
        
    def load_state_dict(self,state_dict):
        """
        Triggered on checkpoint load
        """
        self.resume_index = state_dict["batch_idx"] * state_dict["micro_batch_size"]
 
    def collate_fn(self, x):
        return self._data_collator([y["text"] for y in x])

    def train_dataloader(self):
        """This will be run every epoch."""

        if not hasattr(self, "train_hf_dataset"):
            self.setup_train_hf_dataset()

        # shuffle every epoch
        train_dataset = self.train_hf_dataset.shuffle(
            seed=self.trainer.current_epoch + self.seed
        )

        if self.resume_index is not None:
            self.py_logger.info("Resuming dataloader from example {self.resume_index}")
            # in case we are past the first epoch, may need modulo
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
            prefetch_factor=self.data_args.prefetch_factor,
        )

        self.py_logger.info("Finished loading training data")
        return loader

    def val_dataloader(self):
        if not hasattr(self, "val_hf_dataset"):
            self.setup_val_hf_dataset()

        val_dataset = self.val_hf_dataset

        loader = DataLoader(
            val_dataset,
            batch_size=self.micro_batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.data_args.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=self.data_args.prefetch_factor,
        )

        self.py_logger.info("Finished loading validation data")
        return loader


class OnlineHFDataModule(HFDataModule):
    """
    Datamodule to perform pretraining
    Uses online example packing
    """

    def __init__(
        self,
        tokenizer,
        training_datasets: list,
        validation_datasets: list,
        seed: int,
        micro_batch_size: int,
        data_args,
        data_collator,
        py_logger,
    ):
        super().__init__(
            tokenizer,
            training_datasets,
            validation_datasets,
            seed,
            micro_batch_size,
            data_args,
            data_collator,
            py_logger,
        )

    def get_dataloader_ckpt_folder(self, checkpoint, non_data_filepath=None):
        DATALOADER_CKPT_SPLIT_KEY = "/"

        epoch = checkpoint["epoch"]
        step = checkpoint["global_step"]

        if non_data_filepath:
            pass
            # we are given a checkpoint folder, we will pick the corresponding datafolder
        else:
            # TODO: uses a private method to format, should rely on public methods
            monitor_candidates = self.trainer.checkpoint_callback._monitor_candidates(
                self.trainer
            )
            # always end in backslash so that user can pass l
            non_data_filepath = self.trainer.checkpoint_callback.format_checkpoint_name(
                monitor_candidates
            )

        folder_name = f"dataloader_epoch={epoch}-step={step}"

        # remove trailing '/' if present
        # otherwise this will cause issues with the split
        if non_data_filepath[-1] == "/":
            non_data_filepath = non_data_filepath[:-1]

        data_ckpt_folder = os.path.join(
            DATALOADER_CKPT_SPLIT_KEY.join(
                non_data_filepath.split(DATALOADER_CKPT_SPLIT_KEY)[:-1]
            ),
            folder_name,
        )

        return data_ckpt_folder

    def on_save_checkpoint(self, checkpoint: dict):
        """
        Create folder and save dataloader states there
        """
        data_ckpt_folder = self.get_dataloader_ckpt_folder(checkpoint)

        process_global_rank = th.distributed.get_rank()

        data_ckpt_file = os.path.join(
            data_ckpt_folder, f"rank_{process_global_rank}.pt"
        )

        if self.trainer.is_global_zero:
            try:
                os.makedirs(data_ckpt_folder)
            except FileExistsError as e:
                self.py_logger.warn(
                    "Checkpoint dir already exists, overwriting old dataloader states"
                )
        # workers wait for global 0 to create the folder
        th.distributed.barrier()
        # each worker saves in the folder
        th.save(self._train_dataset.state_dict(), data_ckpt_file)

    def on_load_checkpoint(self, checkpoint: dict):
        """
        Load dataloader states from folder
        """

        data_ckpt_folder = self.get_dataloader_ckpt_folder(
            checkpoint, self.trainer.ckpt_path
        )

        process_global_rank = th.distributed.get_rank()

        data_ckpt_file = os.path.join(
            data_ckpt_folder, f"rank_{process_global_rank}.pt"
        )

        self._train_dataset.load_state_dict(th.load(data_ckpt_file))

    def train_dataloader(self):
        """Construct training dataloader"""

        if not hasattr(self, "train_hf_dataset"):
            self.setup_train_hf_dataset()

        #TODO(colehawk)
        assert self.data_args.num_workers <= 1, "Multiple workers not supported by online packing current dataset"
        self.py_logger.warn(
            "Online example packing is slow, not suitable for production training"
        )
        train_dataset = data.online_packing.OnlinePackedDataset(
            hf_dataset=self.train_hf_dataset,
            tokenizer=self.tokenizer,
            max_tokens_per_example=self.data_args.max_tokens_per_example,
            base_seed=self.seed,
            data_collator=None,
            partition="train",
            process_global_rank=th.distributed.get_rank(),
            detokenize=self.data_args.detokenize,
        )

        # set attribute save state for checkpoint resume
        self._train_dataset = train_dataset

        loader = DataLoader(
            train_dataset,
            batch_size=self.micro_batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.data_args.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=self.data_args.prefetch_factor,
        )

        self.py_logger.info("Finished loading training data")
        return loader

    def val_dataloader(self):

        if not hasattr(self, "val_hf_dataset"):
            self.setup_val_hf_dataset()

        val_dataset = data.online_packing.OnlinePackedDataset(
            hf_dataset=self.val_hf_dataset,
            tokenizer=self.tokenizer,
            max_tokens_per_example=self.data_args.max_tokens_per_example,
            base_seed=self.seed,
            data_collator=None,
            partition="val",
            process_global_rank=th.distributed.get_rank(),
            detokenize=True,  # hack for now, will return text
        )

        #TODO(colehawk)
        assert self.data_args.num_workers <= 1, "Multiple workers not supported by online packing current dataset"
        self.py_logger.warn(
            "Online example packing is slow, not suitable for production training"
        )

        loader = DataLoader(
            val_dataset,
            batch_size=self.micro_batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.data_args.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=self.data_args.prefetch_factor,
        )

        self.py_logger.info("Finished loading validation data")
        return loader
