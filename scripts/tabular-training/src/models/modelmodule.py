"""
Pytorch lightning wrapper for module
"""
import time
import mstar
import pytorch_lightning as pl
import torch as th
import transformers
from mstar.optimizers import FusedAdam
from dataclasses import asdict
from transformers.trainer_pt_utils import get_parameter_names


def get_optimizer(optim_groups, learning_rate, optimizer_cfg):
    optim_cls = {
        "fusedadam": FusedAdam,
    }[optimizer_cfg.optimizer]

    args = [optim_groups]
    kwargs = {
        "lr": learning_rate,
        "eps": optimizer_cfg.adam_epsilon,
        "betas": (optimizer_cfg.adam_beta1, optimizer_cfg.adam_beta2),
    }
    if optimizer_cfg.optimizer in {"fusedadam"}:
        kwargs["adam_w_mode"] = optimizer_cfg.adam_w_mode

    optimizer = optim_cls(*args, **kwargs)
    return optimizer


class PlModel(pl.LightningModule):
    """
    PTL wrapper class for model training
    """

    def __init__(
        self,
        config,
        model,
        py_logger,
        optimizer_cfg,
        scheduler_mult_factor=None,
    ):
        super().__init__()
        self.config = config
        self.model = model
        self.py_logger = py_logger
        self.optimizer_cfg = optimizer_cfg
        self.save_hyperparameters(ignore=["model"])
        self.scheduler_mult_factor = scheduler_mult_factor

    def on_train_start(self):

        if self.scheduler_mult_factor is not None:
            # used to add additional lr scheduler multiplier after checkpoint load
            # otherwise checkpoint load resumes from old scheduler
            # assumes all schedulers are lr lambda schedulers
            self.py_logger.info(
                f"Multiplying all LR schedule lambas by {self.scheduler_mult_factor}"
            )
            self.lr_schedulers().lr_lambdas = [
                lambda x: self.scheduler_mult_factor * fn(x)
                for fn in self.lr_schedulers().lr_lambdas
            ]

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
            batch_size=self.optimizer_cfg.batch_size,
            logger=True,
        )

        return {"loss": loss, "num_loss_tokens": num_loss_tokens}

    def on_train_batch_start(self, batch, batch_idx):
        self._batch_start_time = time.monotonic()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_time = time.monotonic() - self._batch_start_time

        # compute tflops based on batch size the first time
        # assume it does not change over time
        if not hasattr(self, "tflops_per_train_step"):
            self.tflops_per_train_step = mstar.utils.flops_calc.compute_tflops_per_gpu(
                model_type="encoder_decoder",
                sec_per_step=1.0,  # will get actual time during each train-step
                micro_batchsize=self.optimizer_cfg.batch_size,
                activation_checkpointing=self.model.is_gradient_checkpointing,
                vocab_size=self.model.config.vocab_size,
                hidden_size=self.model.config.hidden_size,
                decoder_num_layers=self.model.config.num_decoder_layers,
                encoder_num_layers=self.model.config.num_layers,
                decoder_seq_len=batch["decoder_input_ids"].shape[-1],
                encoder_seq_len=batch["input_ids"].shape[-1],
                use_gated_mlp=getattr(self.model.config, "is_gated_act", False),
            )

        tflops = self.tflops_per_train_step / batch_time
        self.log(
            "training_tflops",
            tflops,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
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
            batch_size=self.optimizer_cfg.batch_size,
            logger=True,
        )
        return {"loss": loss, "num_loss_tokens": num_loss_tokens}

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
            batch_size=self.optimizer_cfg.batch_size,
        )

    def configure_optimizers(self):

        assert self.trainer.max_steps
        # Infer learning rate
        learning_rate = self.optimizer_cfg.base_learning_rate

        # create the optimizer, exclude "bias", "LayerNorm" from decaying
        decay_parameters = get_parameter_names(self.model, [th.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]

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

        optim_groups = [
            {
                "params": params_decay,
                "weight_decay": self.optimizer_cfg.weight_decay,
            },
            {"params": params_nodecay, "weight_decay": 0.0},
        ]

        optimizer = get_optimizer(optim_groups, learning_rate, self.optimizer_cfg)

        scheduler = transformers.optimization.get_scheduler(
            self.optimizer_cfg.lr_scheduler_type,
            optimizer,
            num_warmup_steps=self.optimizer_cfg.warmup_steps,
            num_training_steps=self.trainer.max_steps,
        )

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
