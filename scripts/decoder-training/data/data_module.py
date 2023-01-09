import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from .online_example_packing import ExamplePackDataset

def compute_indices(world_size, current_rank, dataset_length, rng):
    print('computing the index for rank - {}'.format(current_rank))
    all_indices = list(range(0, dataset_length))
    rng.shuffle(all_indices)
    n_sample_per_rank = int(dataset_length / world_size)
    if current_rank < world_size-1:
        indices = all_indices[n_sample_per_rank*current_rank: n_sample_per_rank*(current_rank+1)]
    else:
        indices = all_indices[n_sample_per_rank*current_rank:dataset_length]
    return indices

class PlDataModule(pl.LightningDataModule):
    def __init__(
            self,
            tokenizer,
            training_dataset_path,
            validation_dataset_path,
            seed,
            batch_size,
            data_args,
            py_logger):
        super().__init__()
        self.tokenizer = tokenizer
        self.training_dataset_path = training_dataset_path
        self.validation_dataset_path = validation_dataset_path
        self.seed = seed
        self.batch_size = batch_size
        self.training_data = None
        self.data_args = data_args
        self.py_logger = py_logger
        self.py_logger.info("Using online example packing class to pack the samples")

    def setup(self, stage=None):
        self.py_logger.info("Create memory map\n")
        mmap = pa.memory_map(self.training_dataset_path)
        self.py_logger.info("MMAP Read ALL")
        self._train_dataset = pa.ipc.open_stream(mmap).read_all()

        # Check table has single chunk
        # https://issues.apache.org/jira/browse/ARROW-11989
        assert len(self._train_dataset["text"].chunks) == 1
        #assert len(self._train_dataset["inputs"].chunks) == 1
        valid_mmap = pa.memory_map(self.validation_dataset_path)
        self._valid_dataset = pa.ipc.open_stream(valid_mmap).read_all()
        assert len(self._valid_dataset["text"].chunks) == 1
        #assert len(self._valid_dataset["inputs"].chunks) == 1

    def train_dataloader(self):
        """This will be run every epoch."""
        world_size = torch.distributed.get_world_size()
        process_global_rank = torch.distributed.get_rank()
        seed = self.seed + self.trainer.current_epoch
        rng = np.random.default_rng(seed)
        dataset_indices = compute_indices(world_size, process_global_rank, len(self._train_dataset), rng)

        train_dataset = ExamplePackDataset(self._train_dataset, dataset_indices, self.tokenizer, \
                                     self.batch_size, self.data_args.max_seq_length, self.seed, \
                                     partition='train', max_batch=None)

        self.dl = DataLoader(train_dataset, batch_size=None, collate_fn=None, pin_memory=True, \
                        prefetch_factor=self.data_args.prefetch_factor, num_workers=self.data_args.num_workers)

        self.py_logger.info(f"Finished loading training data")

        # load dataset state from checkpoint here because 'on_load_checkpoint' doesn't work
        if self.data_args.data_state_path:
            data_state = torch.load(self.data_args.data_state_path, map_location="cpu")
            self.dl.dataset.current_epoch = data_state['dataset_state'][process_global_rank][0]
            self.dl.dataset.current_index = data_state['dataset_state'][process_global_rank][1]
            if self.dl.dataset.current_epoch > 0:
                self.dl.dataset._randomize_index()

        return self.dl

    def val_dataloader(self):

        world_size = torch.distributed.get_world_size()
        process_global_rank = torch.distributed.get_rank()
        seed = self.seed + self.trainer.current_epoch
        rng = np.random.default_rng(seed)
        dataset_indices = compute_indices(world_size, process_global_rank, len(self._valid_dataset), rng)

        valid_dataset = ExamplePackDataset(self._valid_dataset, dataset_indices, self.tokenizer, \
                                     self.batch_size, self.data_args.max_seq_length, self.seed, \
                                     partition='valid', max_batch=self.data_args.max_valid_batch)

        dl = DataLoader(valid_dataset, batch_size=None, collate_fn=None, shuffle=False, pin_memory=True, \
                        prefetch_factor=self.data_args.prefetch_factor, num_workers=self.data_args.num_workers)

        self.py_logger.info(f"Finished loading validation data")
        return dl
