import torch
from transformers import AdamW
from transformers.trainer_pt_utils import get_parameter_names
import pytorch_lightning as pl

from mstar.models.gpt2_model import GPT2Config, GPT2LMHeadModel

from apex.optimizers import FusedLAMB, FusedNovoGrad
from mstar.utils import flops_calc
from mstar.optimizers import FusedAdam
from .optimization import get_scheduler
import time
import deepspeed


def get_optimizer(optim_groups, optimizer_cfg):
    optimizer = optimizer_cfg.optimizer.lower()
    optim_cls = {
        "adam": AdamW if optimizer_cfg.adam_w_mode else Adam,
        "fusedadam": FusedAdam,
        "fusedlamb": FusedLAMB,
        "fusednovograd": FusedNovoGrad,
    }[optimizer]

    args = [optim_groups]
    kwargs = {
        "lr": optimizer_cfg.base_learning_rate,
        "eps": optimizer_cfg.adam_epsilon,
        "betas": (optimizer_cfg.adam_beta1, optimizer_cfg.adam_beta2),
    }
    if optimizer in {"fusedadam", "fusedlamb"}:
        kwargs["adam_w_mode"] = optimizer_cfg.adam_w_mode

    optimizer = optim_cls(*args, **kwargs)
    return optimizer

class PlModel(pl.LightningModule):
    def __init__(
            self,
            config,
            tokenizer,
            py_logger,
            optimizer_cfg,
            model_args,
            save_every_n_steps,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.py_logger = py_logger
        self.optimizer_cfg = optimizer_cfg
        self.model_args = model_args
        self.save_every_n_steps = save_every_n_steps
        self._last_batch_end_logged = time.time()
        self.state = None

    # use trainer.num_sanity_val_steps=0 with zero-2d, otherwise the following error will happen
    # https://github.com/microsoft/DeepSpeed/issues/1938
    def setup(self, stage=None):
        #with deepspeed.zero.Init(remote_device=self.trainer.training_type_plugin.remote_device, pin_memory=True,
        #        config=self.trainer.training_type_plugin.config, dtype=torch.bfloat16):
        with deepspeed.zero.Init(remote_device=self.trainer.strategy.remote_device, pin_memory=True,
                config=self.trainer.strategy.config, dtype=torch.bfloat16):
            self.model = GPT2LMHeadModel(self.config)
        self.py_logger.info(f"Loaded GPT model config: {self.config}\n")

    def on_save_checkpoint(self, checkpoint):
        # track the state of the training dataset class to resume training from saved checkpoints
        self.py_logger.info('current state of dataset: {}'.format(self.state))
        checkpoint['dataset_state'] = self.state
        return checkpoint


    def training_step(self, batch, batch_idx):
        # get current_epoch, current_index from batch before forward pass
        current_epoch, current_index = batch.pop('current_epoch'), batch.pop('current_index')

        output = self.model(**batch)
        loss = output.loss

        if batch_idx % self.save_every_n_steps == 0:
            world_size = torch.distributed.get_world_size()
            self.state = [torch.zeros(2, dtype=torch.int64).to(loss.device) for _ in range(world_size)]
            current_state = torch.tensor([current_epoch, current_index])
            # assuming gather happens in order, checked it with trail runs
            torch.distributed.all_gather(self.state, current_state.to(loss.device))

        # average the loss from all devices for logging. we can also consider 'sync_dist' option in
        # pytorch-lightning's self.log in the following.
        num_nonpad_tokens = batch['attention_mask'].count_nonzero()
        # without the '.item()' in loss.item(), the log_loss can be inf/nan due to overflow for
        # lower precision training
        log_loss = loss.item() * num_nonpad_tokens
        torch.distributed.all_reduce(log_loss)
        torch.distributed.all_reduce(num_nonpad_tokens)
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

    def on_train_batch_start(self, batch, batch_idx):
        self._batch_start_time = time.monotonic()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        batch_time = time.monotonic() - self._batch_start_time

        # compute tflops based on batch size the first time
        # assume it does not change over time
        if not hasattr(self, "tflops_per_train_step"):
            self.tflops_per_train_step = flops_calc.compute_tflops_per_gpu(
                model_type="decoder",
                sec_per_step=1.0,  # will get actual time during each train-step
                micro_batchsize=self.optimizer_cfg.batch_size,
                activation_checkpointing=self.model_args.gradient_checkpointing,
                vocab_size=self.config.vocab_size,
                hidden_size=self.config.n_embd,
                decoder_num_layers=self.config.n_layer,
                decoder_seq_len=batch["input_ids"].shape[-1],
                use_gated_mlp=False,
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
        num_nonpad_tokens = batch['attention_mask'].count_nonzero()
        loss = output.loss.item() * num_nonpad_tokens
        return {"loss": loss, "num_nonpad_tokens": num_nonpad_tokens}

    def validation_epoch_end(self, outputs):

        loss = torch.stack([out["loss"] for out in outputs]).sum()
        num_nonpad_tokens = torch.stack(
            [out["num_nonpad_tokens"] for out in outputs]).sum()
        torch.distributed.all_reduce(loss)
        torch.distributed.all_reduce(num_nonpad_tokens)
        loss = loss / num_nonpad_tokens
        self.log(
            "validation_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self):

        # create the optimizer, exclude "bias", "LayerNorm" from decaying
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
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
        optimizer = get_optimizer(optim_groups, self.optimizer_cfg)

        assert self.trainer.max_steps
        scheduler = get_scheduler(
            self.optimizer_cfg.lr_scheduler_type,
            optimizer,
            num_warmup_steps=self.optimizer_cfg.warmup_steps,
            num_training_steps=self.trainer.max_steps,
            min_ratio=self.optimizer_cfg.lr_min_ratio,
            plateau_ratio=self.optimizer_cfg.lr_plateau_ratio,
        )
        self.logger.log_hyperparams(self.optimizer_cfg)
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

