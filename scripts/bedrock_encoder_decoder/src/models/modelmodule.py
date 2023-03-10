"""
Pytorch lightning wrapper for module
"""
import time
import mstar
import hydra
import pytorch_lightning as pl
import torch as th
import transformers
from mstar.optimizers import FusedAdam
from dataclasses import asdict
from transformers.trainer_pt_utils import get_parameter_names

import math
from torch.optim.lr_scheduler import LambdaLR


def get_inverse_sqrt_schedule(
    optimizer, num_warmup_steps, scale_factor=10000, num_constant_steps=0, last_epoch=-1
):
    """Inverse square root learning rate schedule with linear warmup from 0

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
            The total number of training steps.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        scale_factor (:obj:`int`):
            A scaling constant that increases/decreases the decay rate
        num_constant_steps (:obj:`int`):
            The number of steps to hold the max LR constant after warmup
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """

    def lr_lambda(global_step: int):

        if global_step <= num_warmup_steps:
            return global_step / num_warmup_steps
        elif num_warmup_steps < global_step <= num_warmup_steps + num_constant_steps:
            return 1.0
        else:
            # The scale_factor is used to ensure decay is not too rapid.
            # Pure pure 1/sqrt(n) decay leads to 100x decay within 10k steps.
            # scale_factor is present in both the numerator and denominator since
            # the LambdaLR provides a multipler to optimizer.lr which should be 1 at the end of warmup
            return math.sqrt(scale_factor) / math.sqrt(
                scale_factor + global_step - num_warmup_steps - num_constant_steps
            )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class PlModel(pl.LightningModule):
    """
    PTL wrapper class for model training
    """

    def __init__(
        self,
        full_experiment_config,  # to log parameters, should be nested dicts
        model_init_fn,
        py_logger,
        optimizer_cfg,
    ):
        super().__init__()
        self.full_experiment_config = full_experiment_config
        self.model_init_fn = model_init_fn
        self.py_logger = py_logger
        self.optimizer_cfg = optimizer_cfg

    def setup(self, stage: str):
        """
        Set up the module, including model creation
        Args:
            stage: PTL stage train/val/test can be used to induce different 
                    behavior only used for inheritance
        """
        # need to pass trainer into init function
        # required to support state dict load
        self.model = self.model_init_fn(self.trainer)
        # get time here for first iteration at batch 0
        # logged in on_train_batch_end
        self._last_logged_batch_start_time = time.monotonic()

    def training_step(self, batch, batch_idx):

        output = self.model(**batch)
        loss = output.loss

        # exclude tokens that will be excluded from the loss
        num_loss_tokens = (batch["labels"] != -100).count_nonzero()
        # without the '.item()' in loss.item(), the log_loss can
        # be inf/nan due to low-precision+overflow
        log_loss = loss.item() * num_loss_tokens
        th.distributed.all_reduce(log_loss)
        th.distributed.all_reduce(num_loss_tokens)
        log_loss = log_loss / num_loss_tokens

        self.log(
            "training_loss_step",
            log_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=self.optimizer_cfg.micro_batch_size,
            logger=True,
        )

        return {"loss": loss, "num_loss_tokens": num_loss_tokens}

    def calculate_model_tflops(self, batch):
        """
        Assign model TFLOPs as an attribute for easier logging
        Args:
            batch: batch that will be passed into the model, useful for logging
        """
        # compute tflops based on batch size the first time
        # assume it does not change over time
        self.tflops_per_train_step = mstar.utils.flops_calc.compute_tflops_per_gpu(
            model_type="encoder_decoder",
            sec_per_step=1.0,  # will get actual time during each train-step
            micro_batchsize=self.optimizer_cfg.micro_batch_size,
            activation_checkpointing=self.model.is_gradient_checkpointing,
            vocab_size=self.model.config.vocab_size,
            hidden_size=self.model.config.hidden_size,
            decoder_num_layers=self.model.config.num_decoder_layers,
            encoder_num_layers=self.model.config.num_layers,
            decoder_seq_len=batch["decoder_input_ids"].shape[-1],
            encoder_seq_len=batch["input_ids"].shape[-1],
            use_gated_mlp=getattr(self.model.config, "is_gated_act", False),
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        Log time/step and TFLOPS
        Args:
            outputs: outputs of train_step, not used, required for hook
            batch: use batch to get input/output sequence length for TFLOPs
            batch_idx: batch number, not used required for hook
        """

        if batch_idx > 0 and batch_idx % self.trainer.log_every_n_steps == 0:

            # get the time for this iteration
            elapsed_time = time.monotonic() - self._last_logged_batch_start_time
            # start timeer for the next iteration
            self._last_logged_batch_start_time = time.monotonic()

            time_per_step = elapsed_time / self.trainer.log_every_n_steps

            if not hasattr(self, "tflops_per_train_step"):
                # calculate once based on the model
                # then use timing going forward
                self.calculate_model_tflops(batch)

            tflops = self.tflops_per_train_step / time_per_step
            self.log(
                "training_tflops",
                tflops,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                rank_zero_only=True,
            )

            # useful to log this even though PTL provides it in the progressbar
            # PTL logs provide exponential decaying average which is not useful
            # forquick benchmarking, especially for large models
            self.log(
                f"sec/step",
                time_per_step,
                on_step=True,
                prog_bar=True,
                logger=True,
                rank_zero_only=True,
            )

    def validation_step(self, batch, batch_idx):

        output = self.model(**batch)
        # exclude tokens that will be excluded from the loss
        num_loss_tokens = (batch["labels"] != -100).count_nonzero()

        loss = output.loss.item() * num_loss_tokens
        self.log(
            "validation_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.optimizer_cfg.micro_batch_size,
            logger=True,
        )
        return {"loss": loss, "num_loss_tokens": num_loss_tokens}

    def on_train_start(self):

        # override the lambda schedulers
        # default configs do not adjust the schedulers
        self.lr_schedulers().lr_lambdas = [
            lambda x: self.optimizer_cfg.override.mult_factor
            * fn(x + self.optimizer_cfg.override.add_index)
            for fn in self.lr_schedulers().lr_lambdas
        ]

    def validation_epoch_end(self, outputs):

        loss = th.stack([out["loss"] for out in outputs]).sum()
        num_loss_tokens = th.stack([out["num_loss_tokens"] for out in outputs]).sum()
        th.distributed.all_reduce(loss)
        th.distributed.all_reduce(num_loss_tokens)
        loss = loss / num_loss_tokens

        self.log(
            "validation_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.optimizer_cfg.micro_batch_size,
        )

    def configure_optimizers(self):

        # hyperparameter logging needs to occur after ddp launch
        # inside config_optimizers since this occurs after ddp launch
        # use trainer logger which ensures it is mstar logger
        self.trainer.logger.log_hyperparams(self.full_experiment_config)

        # create the optimizer, exclude "bias", "LayerNorm" from decaying
        decay_parameters = get_parameter_names(self.model, [th.nn.LayerNorm])
        # filter out bias
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        # filter out layernorm with a variety of spellings
        decay_parameters = [
            name for name in decay_parameters if "layer_norm" not in name
        ]
        decay_parameters = [
            name for name in decay_parameters if "layernorm" not in name
        ]

        params_decay = [
            p
            for n, p in self.named_parameters()
            if any(nd in n for nd in decay_parameters)
        ]
        params_nodecay = [
            p
            for n, p in self.named_parameters()
            if not any(nd in n for nd in decay_parameters)
        ]

        param_groups = [
            {
                "params": params_decay,
                "weight_decay": self.optimizer_cfg.optimizer.weight_decay,
            },
            {"params": params_nodecay, "weight_decay": 0.0},
        ]

        # need convert="partial" to avoid hydra converting
        # param_groups into an OmegaConf, which breaks optimizer creation
        optimizer = hydra.utils.instantiate(
            self.optimizer_cfg.optimizer, _convert_="partial", params=param_groups
        )

        scheduler = hydra.utils.call(self.optimizer_cfg.scheduler, optimizer=optimizer)
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
