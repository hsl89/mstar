# Preliminaries
import glob
import logging
import os, sys, json
import numpy as np
import pathlib
from dataclasses import dataclass, field, asdict
from typing import Optional
import pyarrow as pa
# torch
import torch as th
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from apex.optimizers import FusedLAMB, FusedNovoGrad

import transformers
from gpt2_model import GPT2Config, GPT2LMHeadModel
from transformers import AdamW, HfArgumentParser

from transformers.trainer_pt_utils import get_parameter_names
import pytorch_lightning as pl
from optimization import SchedulerType, get_scheduler
from data_collator import DataCollatorForGPT

# add paths for loading modules in script and mstar directory
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(curr_path)
sys.path.append(os.path.dirname(curr_path))
sys.path.append(os.path.dirname(os.path.dirname(curr_path)))

from mstar.optimizers import FusedAdam
from mstar.utils.lightning import (
    AWSBatchEnvironment,
    KubeFlowEnvironment,
    AWSBatchProgressBar,
    MStarEKSLogger
)

'''
import warnings
warnings.filterwarnings(
    "ignore", ".*does not have many workers. Consider increasing the value of the `num_workers` argument*"
)

device = 'cuda' if th.cuda.is_available() else 'cpu'
'''


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_type: Optional[str] = field(
        default='GPT2-model',
        metadata={"help": "gpt2-model decoder only architecture"},
    )

    training_dataset: str = field(
        default=None,
        metadata={"help": "Preprocessed arrow file path for training"}
    )
    validation_dataset: str = field(
        default=None,
        metadata={"help": "Preprocessed arrow file path for validation"}
    )
    config_path: str = field(default=None, metadata={"help": "Model config path"})
    tokenizer_path: str = field(
        default=None, metadata={"help": "Pretrained tokenizer path"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )

    state_dict: Optional[str] = field(
        default=None,
        metadata={"help": "path to load new state dict while loading models."},
    )

    gradient_checkpointing: bool = field(
        default=True,
        metadata={
            "help": "Whether to enable gradient checkpointing in the model."
        },
    )

    fused_scaled_masked_softmax: bool = field(
        default=False,
        metadata={"help": "Whether to enable fused softmax in the model."},
    )

    fused_gelu: bool = field(
        default=True,
        metadata={"help": "Whether to enable fused GeLU in the model."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """

    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated."
        },
    )

    train_data_index_path: str = field(
        default=None,
        metadata={"help": "Path to the preprocessed train dataset indices for each gpu"}
    )

    valid_data_index_path: str = field(
        default=None,
        metadata={"help": "Path to the preprocessed validation dataset indices for each gpu"}
    )

    deepspeed_config: Optional[str] = field(
        default=os.path.join(os.path.dirname(__file__), "deepspeed/stage2.json"),
        metadata={"help": "Path to the DeepSpeed config file"},
    )

    save_every_n_steps: Optional[int] = field(
        default=10000,
        metadata={"help": "Save model checkpoints every n steps"},
    )

    num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "Number of workers for dataloader"},
    )

    save_top_k: Optional[int] = field(
        default=5,
        metadata={
            "help": "Save top k model checkpoints if monitor is not None. Default to 1"
                    "Options: 0 to save no checkpoints, n to save n checkpoints, -1 to save all checkpoints."
        },
    )

    length_penalty: Optional[float] = field(
        default=1,
        metadata={"help": "Exponential penalty to the length. 1.0 means no penalty."},
    )

    run_name: Optional[str] = field(
        default=None,
        metadata={"help": "name of the current experiment run, to track in EKS logger"}
    )

class IndexDataset(th.utils.data.Dataset):
    def __init__(self, dataset_indices):
        self.dataset_indices = dataset_indices

    def __getitem__(self, index):
        return self.dataset_indices[index]

    def __len__(self):
        return len(self.dataset_indices)

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
        self.py_logger.info("Using GPT2 data collator")

        self._data_collator = DataCollatorForGPT(
            tokenizer, padding=True, max_length=self.data_args.max_seq_length,
            pad_to_multiple_of=None)

    def setup(self):
        self.py_logger.info("Create memory map\n")
        mmap = pa.memory_map(self.training_dataset_path)
        self.py_logger.info("MMAP Read ALL")
        self._train_dataset = pa.ipc.open_stream(mmap).read_all()

        # Check table has single chunk
        # https://issues.apache.org/jira/browse/ARROW-11989
        assert len(self._train_dataset["text"].chunks) == 1
        valid_mmap = pa.memory_map(self.validation_dataset_path)
        self._valid_dataset = pa.ipc.open_stream(valid_mmap).read_all()
        assert len(self._valid_dataset["text"].chunks) == 1

        '''
        here we read the pre-generated randomized index files for training and validation set. these are split
        into as many number of files as available gpus, so each gpu can read one file.
        '''
        self.train_files = sorted(glob.glob(
                        str(pathlib.Path(self.data_args.train_data_index_path) / "*.mmap"),
                        recursive=True,))
        self.valid_files = sorted(glob.glob(
                        str(pathlib.Path(self.data_args.valid_data_index_path) / "*.mmap"),
                        recursive=True,))

    def pl_collate_fn(self, inputs):
        # truncate huge sequences to a much more manageable length. without this the tokenizer
        # starts hanging with very long input sequences. this is a temporary solution
        # TODO: implement a way to better utilize the long samples for training, possibly by creating
        #  more samples. yet to decide a good strategy to break a sample into multiple samples.
        truncate_length = int(1.05 * self.data_args.max_seq_length)
        for i in range(self.batch_size):
            words = inputs[i].split()
            if len(words) > truncate_length:
                words = words[:truncate_length]
                inputs[i] = ' '.join(words)
        return self._data_collator(inputs)

    def train_collate_fn(self, indices):
        inputs = self._train_dataset.take(indices)["text"].to_pylist()
        return self.pl_collate_fn(inputs)

    def valid_collate_fn(self, indices):
        inputs = self._valid_dataset.take(indices)["text"].to_pylist()
        return self.pl_collate_fn(inputs)

    def train_dataloader(self):
        """This will be run every epoch."""
        process_global_rank = th.distributed.get_rank()
        # read the index in 'r+' mode, as we need to shuffle it later
        dataset_indices = np.memmap(self.train_files[process_global_rank], dtype='uint32', mode='r+')

        # shuffle the indices for every epoch other than 0.
        # the loaded indices are already shuffled

        if self.trainer.current_epoch > 0:
            seed = self.seed + self.trainer.current_epoch
            rng = np.random.default_rng(seed)
            rng.shuffle(dataset_indices)

        train_index_dataset = IndexDataset(dataset_indices)

        dl = DataLoader(
            train_index_dataset,
            batch_size=self.batch_size,
            collate_fn=self.train_collate_fn,
            num_workers=self.data_args.num_workers
        )
        self.py_logger.info(f"Finished loading training data")
        return dl

    def val_dataloader(self):

        process_global_rank = th.distributed.get_rank()
        dataset_indices = np.memmap(self.valid_files[process_global_rank], dtype='uint32', mode='r')

        valid_index_dataset = IndexDataset(dataset_indices)

        dl = DataLoader(
            valid_index_dataset,
            batch_size=self.batch_size,
            collate_fn=self.valid_collate_fn,
            num_workers=self.data_args.num_workers
        )
        self.py_logger.info(f"Finished loading validation data")
        return dl


@dataclass
class OptimizerConfig:
    batch_size: int = 32
    base_learning_rate: float = 3e-4
    weight_decay: float = 0.05
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1e-5
    lr_scheduler_type: SchedulerType = "linear"
    warmup_steps: int = 5000
    lr_min_ratio: float = 0.1
    lr_plateau_ratio: float = 0.1
    seed: int = 1234
    optimizer: str = "Adam"
    adam_w_mode: bool = True

    def __post_init__(self):
        if self.optimizer.lower() not in {
            "adam",
            "fusedadam",
            "fusedlamb",
            "fusednovograd",
        }:
            raise KeyError(
                f"The optimizer type should be one of: Adam, FusedAdam, FusedLAMB, FusedNovoGrad. \
                The current value is {self.optimizer}."
            )

    def get_optimizer(self, optim_groups, learning_rate):
        optimizer = self.optimizer.lower()
        optim_cls = {
            "adam": AdamW if self.adam_w_mode else Adam,
            "fusedadam": FusedAdam,
            "fusedlamb": FusedLAMB,
            "fusednovograd": FusedNovoGrad,
        }[optimizer]

        args = [optim_groups]
        kwargs = {
            "lr": learning_rate,
            "eps": self.adam_epsilon,
            "betas": (self.adam_beta1, self.adam_beta2),
        }
        if optimizer in {"fusedadam", "fusedlamb"}:
            kwargs["adam_w_mode"] = self.adam_w_mode

        optimizer = optim_cls(*args, **kwargs)
        return optimizer


class PlModel(pl.LightningModule):
    def __init__(
            self,
            config,
            tokenizer,
            model,
            py_logger,
            optimizer_cfg: OptimizerConfig,
    ):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.py_logger = py_logger
        self.optimizer_cfg = optimizer_cfg

    def training_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output.loss

        # average the loss from all devices for logging. we can also consider 'sync_dist' option in
        # pytorch-lightning's self.log in the following.
        num_nonpad_tokens = batch['attention_mask'].count_nonzero()
        # without the '.item()' in loss.item(), the log_loss can be inf/nan due to overflow for
        # lower precision training
        log_loss = loss.item() * num_nonpad_tokens
        th.distributed.all_reduce(log_loss)
        th.distributed.all_reduce(num_nonpad_tokens)
        log_loss = log_loss/num_nonpad_tokens

        self.log(
            "training_loss",
            log_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.optimizer_cfg.batch_size,
        )
        return loss


    def validation_step(self, batch, batch_idx):

        output = self.model(**batch)
        num_nonpad_tokens = batch['attention_mask'].count_nonzero()
        loss = output.loss.item() * num_nonpad_tokens
        return {"loss": loss, "num_nonpad_tokens": num_nonpad_tokens}


    def validation_epoch_end(self, outputs):

        loss = th.stack([out["loss"] for out in outputs]).sum()
        num_nonpad_tokens = th.stack(
            [out["num_nonpad_tokens"] for out in outputs]).sum()
        th.distributed.all_reduce(loss)
        th.distributed.all_reduce(num_nonpad_tokens)
        loss = loss / num_nonpad_tokens

        self.log(
            "validation_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        # Infer learning rate
        learning_rate = self.optimizer_cfg.base_learning_rate

        # create the optimizer, exclude "bias", "LayerNorm" from decaying
        decay_parameters = get_parameter_names(self.model, [th.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

        params_decay = [
            p for n, p in self.named_parameters() if any(nd in n for nd in decay_parameters)
        ]
        params_nodecay = [
            p for n, p in self.named_parameters() if not any(nd in n for nd in decay_parameters)
        ]

        optim_groups = [
            {
                "params": params_decay,
                "weight_decay": self.optimizer_cfg.weight_decay,
            },
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = self.optimizer_cfg.get_optimizer(optim_groups, learning_rate)

        assert self.trainer.max_steps
        scheduler = get_scheduler(
            self.optimizer_cfg.lr_scheduler_type,
            optimizer,
            num_warmup_steps=self.optimizer_cfg.warmup_steps,
            num_training_steps=self.trainer.max_steps,
            min_ratio=self.optimizer_cfg.lr_min_ratio,
            plateau_ratio=self.optimizer_cfg.lr_plateau_ratio,
        )
        self.logger.log_hyperparams(asdict(self.optimizer_cfg))
        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "reduce_on_plateau": False,
                    "monitor": "validation_loss",
                }
            ],
        )


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OptimizerConfig))
    parser = pl.Trainer.add_argparse_args(parser)
    (
        model_args,
        data_args,
        optimizer_cfg,
        trainer_args,
    ) = parser.parse_args_into_dataclasses()
    logger.info(f"model_args: {model_args}\n\n")
    logger.info(f"data_args: {data_args}\n\n")
    logger.info(f"optimizer_cfg: {optimizer_cfg}\n\n")
    logger.info(f"trainer_args: {trainer_args}\n\n")

    # Set env variable for using tokenizer in multiple process dataloader 
    if data_args.num_workers > 0:
        os.environ['TOKENIZERS_PARALLELISM'] = "true"

    # Set seed before initializing model.
    pl.utilities.seed.seed_everything(optimizer_cfg.seed)

    # Load tokenizer and model

    #tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.tokenizer_path)
    tokenizer = transformers.GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "max_length": data_args.max_seq_length,
        "length_penalty": data_args.length_penalty
    }

    with open(model_args.config_path) as infile:
        model_config = json.load(infile)
        logger.info(f"Loaded config from {model_args.config_path}:\n{model_config}")
    model_config.update(
        {
            "vocab_size": len(tokenizer),
            "fused_scaled_masked_softmax": model_args.fused_scaled_masked_softmax,
            "fused_gelu": model_args.fused_gelu,
            "gradient_checkpointing": model_args.gradient_checkpointing
        }
    )

    config = GPT2Config(**{**model_config, **config_kwargs})
    model = GPT2LMHeadModel(config)
    logger.info(f"Loaded GPT model config: {config}\n")


    if model_args.state_dict:
        state_dict = th.load(model_args.state_dict, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if "module" in state_dict:
            state_dict = state_dict["module"]
        unwrapped_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module"):  # deepspeed zero_to_fp32.py
                new_key = ".".join(key.split(".")[2:])
            else:
                new_key = ".".join(key.split(".")[1:])
            unwrapped_state_dict[new_key] = value
        model.load_state_dict(unwrapped_state_dict)
        logger.info(
            f"loaded the model parameters from the ckpt file {model_args.state_dict}"
        )


    # Infer number of nodes
    if "AWS_BATCH_JOB_NUM_NODES" in os.environ:
        num_nodes = trainer_args.num_nodes
        batch_num_nodes = int(os.environ["AWS_BATCH_JOB_NUM_NODES"])
        if num_nodes != batch_num_nodes:
            logging.warning(
                f"--trainer.num_nodes={num_nodes} != "
                f"$AWS_BATCH_JOB_NUM_NODES={batch_num_nodes}. "
                f"Setting --trainer.num_nodes={batch_num_nodes}!"
            )
            trainer_args.num_nodes = batch_num_nodes

    if "KUBERNETES_SERVICE_HOST" in os.environ:
        num_nodes = trainer_args.num_nodes
        kubeflow_num_nodes = int(os.environ["NUM_NODES"])
        if num_nodes != kubeflow_num_nodes:
            logging.warning(
                f"--trainer.num_nodes={num_nodes} != "
                f"$NUM_NODES={kubeflow_num_nodes}. "
                f"Setting --trainer.num_nodes={kubeflow_num_nodes}!"
            )
            trainer_args.num_nodes = kubeflow_num_nodes

    training_dataset = model_args.training_dataset
    validation_dataset = model_args.validation_dataset
    data_module = PlDataModule(
        tokenizer=tokenizer,
        training_dataset_path=training_dataset,
        validation_dataset_path=validation_dataset,
        seed=optimizer_cfg.seed,
        batch_size=optimizer_cfg.batch_size,
        data_args=data_args,
        py_logger=logger,
    )

    model_module = PlModel(
        config=config,
        tokenizer=tokenizer,
        model=model,
        py_logger=logger,
        optimizer_cfg=optimizer_cfg,
    )

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            every_n_train_steps=data_args.save_every_n_steps,
            save_top_k=data_args.save_top_k,
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
        if not data_args.run_name:
            run_name = '{}-input-{}'.format(model_args.model_type, data_args.max_seq_length)
        else:
            run_name = data_args.run_name
        mstar_logger = MStarEKSLogger(experiment_name="gpt2-experiment",
                                      run_name=run_name, tags={"mode": "Training"})
    if trainer_args.gpus == -1:
        trainer_args.gpus = th.cuda.device_count()

    if trainer_args.accelerator == "deepspeed":
        print('using deepspeed ...')
        plugins.append(
            pl.plugins.training_type.DeepSpeedPlugin(
                config=data_args.deepspeed_config,
                remote_device=None  # Initialize directly on GPUs instead of CPU (ZeRO-3)
            )
        )

    trainer = pl.Trainer.from_argparse_args(trainer_args,
                                            callbacks=callbacks,
                                            # Resample each epoch
                                            reload_dataloaders_every_epoch=True,
                                            plugins=plugins,
                                            logger=mstar_logger)

    logger.info(f"*********** data module set up ***********\n\n")
    data_module.setup()
    logger.info(f"*********** start training ***********\n\n")
    trainer.fit(model=model_module, datamodule=data_module, )

    if trainer.is_global_zero or trainer_args.accelerator == "deepspeed":
        # save_checkpoint on all rank with deepspeed to avoid hang
        # https://github.com/microsoft/DeepSpeed/issues/797
        save_path = os.path.join(trainer.default_root_dir, "last.ckpt")
        logger.info(f"Saving model to {save_path}")
        trainer.save_checkpoint(save_path, weights_only=True)
        logger.info("Finished saving")
        if trainer_args.accelerator == "deepspeed":
            # Avoid checkpoint corruption if node 0 exits earlier than other
            # nodes triggering termination of other nodes
            th.distributed.barrier()

if __name__ == "__main__":
    """process the data per batch; partition data indices files. 
        Set --replace_sampler_ddp False using SequentialSampler
    """
    main()