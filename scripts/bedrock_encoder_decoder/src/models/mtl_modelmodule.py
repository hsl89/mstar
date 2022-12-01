"""
Pytorch lightning wrapper for module
"""
import time
import hydra
import mstar
import pytorch_lightning as pl
import torch as th
import transformers
from mstar.optimizers import FusedAdam
from dataclasses import asdict
from transformers.trainer_pt_utils import get_parameter_names

import math
from torch.optim.lr_scheduler import LambdaLR

from models.modelmodule import PlModel

from collators import t5_collator
import numpy as np

 
class PlModelMTL(PlModel):
    """
    PTL wrapper class for model training
    """

    def __init__(
        self,
        full_experiment_config,
        model_init_fn,
        py_logger,
        optimizer_cfg,
        unlabeled_batch_size = 0,
        labeled_batch_size = 1,
        tokenizer = None,
        val_loss_names = ['labeled_val_loss', 'validation_loss']
    ):
       
        super().__init__(
            full_experiment_config=full_experiment_config,  # pass full cfg over for easier logging
            model_init_fn=model_init_fn,
            py_logger=py_logger,
            optimizer_cfg=optimizer_cfg,
        )
        self.unlabeled_batch_size = unlabeled_batch_size
        self.labeled_batch_size = labeled_batch_size
        if self.unlabeled_batch_size == 0:  #Only P3 
            self.val_loss_names = ['labeled_val_loss']
        elif self.labeled_batch_size == 0: #Only PILE
            self.val_loss_names = ['validation_loss']
        else:
            self.val_loss_names = val_loss_names
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.decoder_start_token_id = self.full_experiment_config.model.decoder_start_token_id
        self.tensor_device = None


    def setup(self, stage):

        self.model = self.model_init_fn()
        # always use gradient checkpointing
        #self.model.gradient_checkpointing_enable()
        # setup the random number generator
        self.process_global_rank = th.distributed.get_rank()
        self.rng = np.random.default_rng(self.optimizer_cfg.seed+self.process_global_rank)


    def concatenate_batches(self, batches: list):
        """Takes a list of batches and concatenates them
           Assumption: first batch is for labeled P3 and second batch is for unlabeled PILE 
        """
        if len(batches) > 1:

            #Handle corner-case where pile batch can be empty due to CLM collator
            #See this: https://gitlab.aws.dev/mstar/mstar/-/blob/mtl_bedrock/scripts/bedrock_encoder_decoder/src/collators/t5_collator.py#L746 
            if batches[1] == {}:
                #Return only P3 batch
                return batches[0] 

            batch = {}
            for k in batches[0]:
                p3, pile = batches[0][k], batches[1][k]
                assert (pile.shape[-1] == p3.shape[-1]), 'P3 and PILE should have the same sequence length'
                batch[k] = th.cat([pile, p3])
            return batch
        else:
            return batches[0]

    def downsample_batch(self, batch, batch_size):
        all_indices = list(range(batch_size))
        self.rng.shuffle(all_indices)
        for key, val in batch.items():
            batch[key] = val[all_indices[0:self.optimizer_cfg.downsample_batch_to]]
        return batch

    def training_step(self, batch, batch_idx):

        #Added to support multiple data-loaders
        if type(batch)==list:
            batch = self.concatenate_batches(batch)

        #Downsample batch if self.optimizer_cfg.downsample_batch_to is set to value smaller than batch_size
        batch_size = len(batch[list(batch.keys())[0]])
        if getattr(self.optimizer_cfg, "downsample_batch_to", False):
            if self.optimizer_cfg.downsample_batch_to < batch_size:
                batch = self.downsample_batch(batch, batch_size)

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


    def on_train_batch_end(self, outputs, batch, batch_idx):

        #Added to support multiple data-loaders
        if type(batch)==list:
            batch = self.concatenate_batches(batch)

        batch_time = time.monotonic() - self._batch_start_time

        # compute tflops based on batch size the first time
        # assume it does not change over time
        if not hasattr(self, "tflops_per_train_step"):
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

        tflops = self.tflops_per_train_step / batch_time
        self.log(
            "training_tflops",
            tflops,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )

    def create_dummy_batch(self, device) -> dict():
        """
        Returns a dummy batch of data
        """
        
        #label dimension is 1 x max_output_lenght
        #Fix the labels 0 (results in a large loss-value, therefore loss-value should be ignored)
        labels = 0*th.ones(1, self.full_experiment_config.data.max_output_length, dtype=th.int64)
        
        #Only padding tokens in input
        #label dimension is 1 x max_seq_lenght
        input_ids = self.pad_token_id*th.ones(1, self.full_experiment_config.data.max_seq_length, dtype=th.int64)

        decoder_input_ids = t5_collator.shift_tokens_right(labels, self.pad_token_id, self.decoder_start_token_id)

        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
        }
        
        #Create attention mask to ignore the padding tokens in the input
        batch["attention_mask"] = np.where(batch["input_ids"] == self.pad_token_id, 0, 1)

        for key, val in batch.items():
            batch[key] = th.LongTensor(val).to(device)
        return batch


    def is_positive(self, a: th.Tensor) -> bool:
        """
        Returns True if all elements of the tensor are greater-than-equal-to zero, otherwise False
        """

        zero_tensor = th.zeros(a.shape, device=self.tensor_device)
        if th.all(th.ge(a, zero_tensor)):
            return True
        else:
            return False 


    def validation_step(self, batch, batch_idx, *args):

        empty_batch = False

        #Handle corner-case where pile batch can be empty due to CLM collator
        #See this: https://gitlab.aws.dev/mstar/mstar/-/blob/mtl_bedrock/scripts/bedrock_encoder_decoder/src/collators/t5_collator.py#L746 
        #Empty unlabeled data or labeled data
        if batch == {}:
            self.py_logger.info(f"Validation_Step: corner case where data batch is empty, Creating dummy batch and forcing loss to be close-to zero")
            empty_batch = True
            batch = self.create_dummy_batch(device=self.tensor_device)

        else:
            #Assumption: Labeled data is not empty and it comes first for validation (data-loader:0)
            self.tensor_device = batch["labels"].device

        output = self.model(**batch)
        # exclude tokens that will be excluded from the loss
        num_loss_tokens = (batch["labels"] != -100).count_nonzero()

        #Forcing num_loss_tokens (and therefore the loss) to be negative one for empty batches so that it can be ignored later
        if empty_batch:
            num_loss_tokens = th.neg(num_loss_tokens/num_loss_tokens) 
        
        loss = output.loss.item() * num_loss_tokens
        
        if self.is_positive(loss):
            val_idx = 0 if args == () else args[0] #index for selecting the correct name
            self.log(
                self.val_loss_names[val_idx],  #Selecting the corresponding loss-name
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                batch_size=self.optimizer_cfg.micro_batch_size,
                logger=True,
            )
        return {"loss": loss, "num_loss_tokens": num_loss_tokens}

    def validation_epoch_end(self, outputs):
        
        if self.unlabeled_batch_size==0 or self.labeled_batch_size==0:
            outputs = [outputs] #Making a list to ensure that the for-loop below works

        try:
            #Outputs is list with labeled_outputs and unlabeled_outputs
            if len(self.val_loss_names) != len(outputs):
                raise ValueError(f'Validation Epoch End: Length mismatch for loss names, got {len(self.val_loss_names)} vs {len(outputs)}')
                
            for val_loss_name, outputs in zip(self.val_loss_names, outputs):
                loss = th.stack([out["loss"] for out in outputs if self.is_positive(out["loss"])]).sum()
                num_loss_tokens = th.stack([out["num_loss_tokens"] for out in outputs if self.is_positive(out["num_loss_tokens"])]).sum()
                th.distributed.all_reduce(loss)
                th.distributed.all_reduce(num_loss_tokens)

                loss = loss / num_loss_tokens

                self.log(
                    val_loss_name,
                    loss,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=self.optimizer_cfg.micro_batch_size,
                )

        except ValueError as error:
            print (str(error))
            print ('Validation Epoch End: Observed failure, logging -10 as val_loss')
            loss = th.tensor(-10) #Loss assigned to an arbitray value so that we can observe this in the ML-flow log as well
            self.log(
                self.val_loss_names[0],
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=self.optimizer_cfg.micro_batch_size,
            )


    def configure_optimizers(self):

        # save to log config here because it takes place after setup/distributed launch
        # logging before setup logs on all processes and causes duplicated
        self.logger.log_hyperparams(self.full_experiment_config)
        # full_experiment_config)

        assert self.trainer.max_steps

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
            {"params": params_decay, "weight_decay": self.optimizer_cfg.optimizer.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]

        #need convert="partial" to avoid hydra converting 
        #param_groups into an OmegaConf, which breaks optimizer creation 
        optimizer = hydra.utils.instantiate(self.full_experiment_config.optimization.optimizer, _convert_="partial", params=param_groups)

        scheduler = hydra.utils.call(self.full_experiment_config.optimization.scheduler, optimizer=optimizer)

        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                    "reduce_on_plateau": False,
                    "monitor": self.val_loss_names[0], #Always monitor P3 val loss
                }
            ],
        )

