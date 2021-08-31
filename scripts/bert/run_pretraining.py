"""Pretraining on Code"""
import functools
import glob
import logging
import math
import multiprocessing
import os
import pathlib
import random
import sys
import typing
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute
import pyarrow.dataset
import pyarrow.feather
import pytorch_lightning as pl
import torch as th
import torchmetrics
import transformers
from numpy.testing import assert_allclose
from pytorch_lightning.utilities.cli import LightningCLI
from pytorch_lightning.utilities.cloud_io import load as pl_load
from torch.nn import functional as F
from torch.utils.data import DataLoader

import mstar
from mstar.utils.data import MLMS3DataConfig


def identity_collate(x):  # Can't pickle lambda on Py3.6
    return x


@dataclass
class NumpyDataset(th.utils.data.Dataset):
    array: np.ndarray

    def __getitem__(self, index):
        return self.array[index]

    def __len__(self):
        return len(self.array)


def ner_collate_fn(indices, *, tbl):
    batch = tbl.take(indices).to_pydict()
    input_id = th.nn.utils.rnn.pad_sequence([th.tensor(ele) for ele in batch['tokens']],
                                            batch_first=True)

    segment_id = th.zeros_like(input_id)
    valid_length = th.tensor(batch['validlength'])
    mlm_positions = batch['mlmpositions1']
    # Masked positions with respect to flattened contextual_embedding
    # (batch_size * seq_length, units)
    seq_length = input_id.shape[1]
    mlm_positions = [np.array(pos) + seq_length * i for i, pos in enumerate(mlm_positions)]
    mlm_positions = th.tensor(np.concatenate(mlm_positions).astype(np.int64))
    mlm_labels = th.tensor(np.concatenate(batch['mlmlabels']).astype(np.int64))
    ner_labels = th.tensor(np.concatenate(batch['seqtagging_labels']).astype(np.int64))
    ner_positions = [np.arange(vlen) + seq_length * i for i, vlen in enumerate(valid_length)]
    ner_positions = th.tensor(np.concatenate(ner_positions).astype(np.int64))
    return input_id, segment_id, valid_length, mlm_positions, mlm_labels, ner_positions, ner_labels


class MultiTaskDataModule(pl.LightningDataModule):
    def __init__(self, mlm_cfg: MLMS3DataConfig = MLMS3DataConfig(),
                 ner_dir: str = "./ner_feather",
                 num_workers=1, prefetch_factor=32, single_task_mode: bool = True):
        super().__init__()
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self._mlm_cfg = mlm_cfg
        self._ner_dir = ner_dir
        self._single_task_mode = single_task_mode
        self._train_paths, self._eval_paths = mstar.utils.data.get_train_and_eval_paths(mlm_cfg)

    def train_dataloader(self):
        mlm_dataloader = DataLoader(
            mstar.utils.data.DistributedMLM(self._train_paths, self._mlm_cfg, infinite=True),
            batch_size=1, collate_fn=identity_collate, pin_memory=True)
        data = {'mlm': mlm_dataloader}

        if not self._single_task_mode:
            ner_files = sorted(
                glob.glob(
                    str(pathlib.Path(self._ner_dir) / "**" / "*feather"),
                    recursive=True,
                ) + glob.glob(str(pathlib.Path(self._ner_dir) / "*feather")))
            if not ner_files:
                logging.warning("No MLM+NER data!")
                sys.exit(1)
            ds = pa.dataset.dataset(ner_files, format="feather")
            # Without combining chunks tbl.take is 1000x slower
            tbl = ds.to_table().combine_chunks()
            ner_dataloader = DataLoader(
                NumpyDataset(np.arange(len(tbl))),
                collate_fn=functools.partial(ner_collate_fn, tbl=tbl),
                batch_size=self._mlm_cfg.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            data["ner"] = ner_dataloader

        return data

    def val_dataloader(self):
        cfg = MLMS3DataConfig()
        cfg.replace(self._mlm_cfg)
        cfg.use_nsp = False
        mlm_loader = DataLoader(
            mstar.utils.data.DistributedMLM(
                self._eval_paths, cfg,
                infinite=False, eager=True, shuffle=False, keep_last_batch=True),
            batch_size=1,
            collate_fn=identity_collate, pin_memory=True)
        return mlm_loader

    def test_dataloader(self):
        raise NotImplementedError


class CountMetric(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("count", default=th.tensor([0], dtype=th.int64), dist_reduce_fx="sum")

    def update(self, count: th.Tensor):
        self.count += count

    def compute(self):
        return self.count.float()


class LossMetric(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("loss", default=th.tensor([0], dtype=th.float64), dist_reduce_fx="sum")
        self.add_state("count", default=th.tensor([0], dtype=th.float64), dist_reduce_fx="sum")

    def update(self, loss: th.Tensor, count: th.Tensor):
        self.loss += loss.double()
        self.count += count.double()

    def compute(self):
        return self.loss.double() / self.count.double()


@dataclass
class BertModelConfig:
    vocab_size: int = 119547
    units: int = 768
    hidden_size: int = 3072
    max_length: int = 512
    num_heads: int = 12
    num_layers: int = 12
    pos_embed_type: str = 'learned'
    activation: str = 'gelu'
    pre_norm: bool = True
    layer_norm_eps: float = 1E-12
    num_token_types: int = 2
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layout: str = 'NT'
    compute_layout: str = 'auto'


@dataclass
class OptimizerConfig:
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    betas: typing.Tuple[float, float] = (0.9, 0.95)
    batch_size: typing.Optional[int] = None  # Bound to mlm_cfg.batch_size
    base_batch_size: int = 256
    base_learning_rate: float = 1e-4
    base_warmup_ratio: float = 0.003125


@dataclass
class NERConfig:
    vocab_size: int = 81


class MultiTaskBert(pl.LightningModule):
    """MultiTaskBert consists of a BERT backbone and task specific heads.

    Supported tasks must be specified at construction. At forward pass, the
    task name of the supplied batch must be specified.

    """
    def __init__(self, *, bert_cfg: BertModelConfig = BertModelConfig(),
                 optimizer_cfg: OptimizerConfig = OptimizerConfig(),
                 ner_cfg: NERConfig = NERConfig(), base_max_steps: int = 900000,
                 single_task_mode: bool = False, phase1_ckpt_path: str = None):
        super().__init__()
        # auto creates self.hparams from the method signature
        self.save_hyperparameters()

        # in lightning the "config" is hparams (for hyperparameters)
        self.base_max_steps = base_max_steps
        self.single_task_mode = single_task_mode
        self.bert_cfg = bert_cfg
        self.optimizer_cfg = optimizer_cfg

        # Model
        self.bert = mstar.models.bert.BertModel(
            vocab_size=bert_cfg.vocab_size, units=bert_cfg.units, hidden_size=bert_cfg.hidden_size,
            num_layers=bert_cfg.num_layers, num_heads=bert_cfg.num_heads,
            max_length=bert_cfg.max_length, hidden_dropout_prob=bert_cfg.hidden_dropout_prob,
            attention_dropout_prob=bert_cfg.attention_dropout_prob,
            num_token_types=bert_cfg.num_token_types, pos_embed_type=bert_cfg.pos_embed_type,
            activation=bert_cfg.activation, layer_norm_eps=bert_cfg.layer_norm_eps, use_pooler=True,
            layout=bert_cfg.layout, compute_layout=bert_cfg.compute_layout,
            pre_norm=bert_cfg.pre_norm)

        self.quickthought = th.nn.Sequential(
            th.nn.Linear(out_features=bert_cfg.units, in_features=bert_cfg.units),
            mstar.layers.get_activation(bert_cfg.activation),
            th.nn.LayerNorm(bert_cfg.units, eps=bert_cfg.layer_norm_eps))

        self.mlm_decoder = th.nn.Sequential(
            th.nn.Linear(out_features=bert_cfg.units, in_features=bert_cfg.units),
            mstar.layers.get_activation(bert_cfg.activation),
            th.nn.LayerNorm(bert_cfg.units, eps=bert_cfg.layer_norm_eps),
            th.nn.Linear(out_features=bert_cfg.vocab_size, in_features=bert_cfg.units))

        self.nsp_decoder = th.nn.Linear(out_features=2, in_features=bert_cfg.units)

        self.num_tokens_metric = CountMetric(dist_sync_on_step=True)
        self.mlm_loss_metric = LossMetric(dist_sync_on_step=True)
        self.mlm_val_loss_metric = LossMetric(dist_sync_on_step=True)
        self.nsp_loss_metric = LossMetric(dist_sync_on_step=True)
        self.mlm_acc_metric = torchmetrics.Accuracy(dist_sync_on_step=True)
        self.mlm_val_acc_metric = torchmetrics.Accuracy(dist_sync_on_step=True)
        self.nsp_acc_metric = torchmetrics.Accuracy(dist_sync_on_step=True)

        if not self.single_task_mode:
            self.ner_decoder = th.nn.Sequential(
                th.nn.Linear(out_features=bert_cfg.units, in_features=bert_cfg.units),
                mstar.layers.get_activation(bert_cfg.activation),
                th.nn.LayerNorm(bert_cfg.units, eps=bert_cfg.layer_norm_eps),
                th.nn.Linear(out_features=ner_cfg.vocab_size, in_features=bert_cfg.units))
            self.ner_loss_metric = LossMetric(dist_sync_on_step=True)
            self.ner_acc_metric = torchmetrics.Accuracy(dist_sync_on_step=True)

        # Initialization
        if phase1_ckpt_path is None:
            self.apply(mstar.models.bert.init_weights)
        elif phase1_ckpt_path.endswith('.ckpt'):
            ckpt = pl_load(phase1_ckpt_path)
            self.load_state_dict(ckpt['state_dict'])
        elif phase1_ckpt_path.endswith('.params'):  # GluonNLPv1 / MXNet 2 checkpoint
            assert not bert_cfg.pre_norm
            import mxnet as mx
            mx.npx.set_np()
            mx_params = mx.npx.load(phase1_ckpt_path)
            mx_params = {k.replace('.beta', '.bias'): v for k, v in mx_params.items()}
            mx_params = {k.replace('.gamma', '.weight'): v for k, v in mx_params.items()}
            mx_params = {
                k.replace('token_pos_embed._embed.weight', 'token_pos_embed.weight'): v
                for k, v in mx_params.items()
            }
            backbone_params = {
                k[15:]: th.Tensor(v.asnumpy())
                for k, v in mx_params.items() if k.startswith('backbone')
            }
            mlm_params = {
                k[12:]: th.Tensor(v.asnumpy())
                for k, v in mx_params.items() if k.startswith('mlm')
            }
            assert set(self.bert.state_dict().keys()) == set(backbone_params.keys())
            assert set(self.mlm_decoder.state_dict().keys()) == set(mlm_params.keys())
            self.bert.load_state_dict(backbone_params)
            self.mlm_decoder.load_state_dict(mlm_params)
        else:
            raise ValueError(f'Unexpected phase1_ckpt: {phase1_ckpt_path}')

    def configure_optimizers(self):
        # Infer learning rate
        global_batch_size = (self.trainer.num_nodes * self.trainer.gpus *
                             self.optimizer_cfg.batch_size)
        factor = global_batch_size / self.optimizer_cfg.base_batch_size
        learning_rate = self.optimizer_cfg.base_learning_rate * math.sqrt(
            factor) if factor > 1 else self.optimizer_cfg.base_learning_rate
        warmup_ratio = (self.optimizer_cfg.base_warmup_ratio *
                        factor if factor > 1 else self.optimizer_cfg.base_learning_rate)
        if self.trainer.is_global_zero:
            print(f'Inferred LR as {learning_rate:.4f} based on sqrt scaling rule '
                  f'(BS scale {factor:.2f}x).')
            print(f'Inferred warmup-ratio as {warmup_ratio:.4f} based on sqrt scaling rule '
                  f'(BS scale {factor:.2f}x).')

        # create the optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [
            p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)
        ]
        params_nodecay = [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)]
        optim_groups = [
            {
                "params": params_decay,
                "weight_decay": self.hparams.optimizer_cfg.weight_decay
            },
            {
                "params": params_nodecay,
                "weight_decay": 0.0
            },
        ]
        if self.optimizer_cfg.optimizer.lower() == "lamb":
            from deepspeed.ops.lamb import FusedLamb
            optimizer = FusedLamb(optim_groups, lr=learning_rate,
                                  betas=self.hparams.optimizer_cfg.betas)
        elif self.optimizer_cfg.optimizer.lower() == "adamw":
            optimizer = mstar.optimizers.FusedAdam(optim_groups, lr=learning_rate,
                                                   betas=self.hparams.optimizer_cfg.betas)
        else:
            raise ValueError

        # configure the learning rate schedule
        assert self.base_max_steps * warmup_ratio < self.trainer.max_steps
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_ratio * self.base_max_steps,
            num_training_steps=self.trainer.max_steps)

        return ([optimizer], [{
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
            'reduce_on_plateau': False,
            'monitor': 'val_loss',
        }])

    def forward(self, data, valid_length, task=None):
        # TODO
        pass

    def training_step(self, batch, batch_idx):
        mlm_loss, nsp_loss, ner_loss = 0, 0, 0
        if random.random() < 0.99 or self.single_task_mode:
            input_id, segment_id, valid_length, mlm_positions, mlm_labels, nsp_labels = batch[
                'mlm'][0]
            mlm_features, pooled_out = self.bert(input_id, segment_id, valid_length)
            if self.bert_cfg.layout == 'NT':
                mlm_features = mlm_features.flatten(0, 1)[mlm_positions]
            else:
                mlm_features = th.transpose(mlm_features, 0, 1).flatten(0, 1)[mlm_positions]
            mlm_scores = self.mlm_decoder(mlm_features)
            nsp_scores = self.nsp_decoder(pooled_out)
            mlm_loss = F.cross_entropy(mlm_scores, mlm_labels)
            nsp_loss = F.cross_entropy(nsp_scores, nsp_labels)

            self.mlm_acc_metric(mlm_scores.argmax(dim=1), mlm_labels)
            self.mlm_loss_metric(mlm_loss * len(mlm_labels),
                                 th.full((1, ), len(mlm_labels), device=self.device))
            self.nsp_acc_metric(nsp_scores.argmax(dim=1), nsp_labels)
            self.nsp_loss_metric(nsp_loss * len(nsp_labels),
                                 th.full((1, ), len(nsp_labels), device=self.device))

            self.log('mlm_loss', self.mlm_loss_metric, on_step=True, on_epoch=False, prog_bar=True)
            self.log('nsp_loss', self.nsp_loss_metric, on_step=True, on_epoch=False, prog_bar=True)
            self.log('mlm_acc', self.mlm_acc_metric, on_step=True, on_epoch=False, prog_bar=True)
            self.log('nsp_acc', self.nsp_acc_metric, on_step=True, on_epoch=False, prog_bar=True)
        else:
            (input_id, segment_id, valid_length, mlm_positions, mlm_labels, ner_positions,
             ner_labels) = batch['ner']
            ner_features, pooled_out = self.bert(input_id, segment_id, valid_length)
            if self.bert_cfg.layout == 'NT':
                ner_features = ner_features.flatten(0, 1)[ner_positions]
            else:
                ner_features = th.transpose(ner_features, 0, 1)\
                                 .flatten(0, 1)[ner_positions]
            ner_scores = self.ner_decoder(ner_features)
            ner_loss = F.cross_entropy(ner_scores, ner_labels)

            self.ner_acc_metric(ner_scores.argmax(dim=1), ner_labels)
            self.ner_loss_metric(ner_loss * len(ner_labels),
                                 th.full((1, ), len(ner_labels), device=self.device))

            self.log('ner_loss', self.ner_loss_metric, on_step=True, on_epoch=False, prog_bar=False,
                     logger=True)
            self.log('ner_acc', self.ner_acc_metric, on_step=True, on_epoch=False, prog_bar=False,
                     logger=True)

        self.num_tokens_metric(valid_length.sum())  # TODO throughput
        self.log('num_tokens', self.num_tokens_metric, on_step=True, on_epoch=False, prog_bar=True)

        return mlm_loss + nsp_loss + ner_loss

    def validation_step(self, batch, batch_idx):
        input_id, segment_id, valid_length, mlm_positions, mlm_labels = batch[0]
        mlm_features, pooled_out = self.bert(input_id, segment_id, valid_length)
        if self.bert_cfg.layout == 'NT':
            mlm_features = mlm_features.flatten(0, 1)[mlm_positions]
        else:
            mlm_features = th.transpose(mlm_features, 0, 1).flatten(0, 1)[mlm_positions]
        mlm_scores = self.mlm_decoder(mlm_features)
        mlm_loss = F.cross_entropy(mlm_scores, mlm_labels)

        self.mlm_val_acc_metric(mlm_scores.argmax(dim=1), mlm_labels)
        self.mlm_val_loss_metric(mlm_loss * len(mlm_labels),
                                 th.full((1, ), len(mlm_labels), device=self.device))

    def validation_epoch_end(self, val_outs):
        self.log('mlm_val_loss', self.mlm_val_loss_metric.compute(), on_step=False, on_epoch=True,
                 prog_bar=True)
        self.log('mlm_val_acc', self.mlm_val_acc_metric.compute(), on_step=False, on_epoch=True,
                 prog_bar=True)
        self.mlm_val_loss_metric.reset()
        self.mlm_val_acc_metric.reset()

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def test_epoch_end(self, test_outs):
        raise NotImplementedError


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument(
            '--validate_only', action="store_true",
            help='See https://github.com/PyTorchLightning/pytorch-lightning/discussions/7226')
        parser.add_argument('--export_to_mxnet', action="store_true",
                            help='Export trained model for MXNet.')
        parser.add_argument('--bf16', action="store_true", help='Use bfloat16 format for training. '
                            'Deepspeed is not supported.')

        # Link arguments
        parser.link_arguments('data.mlm_cfg.batch_size', 'model.optimizer_cfg.batch_size')

    def before_instantiate_classes(self) -> None:
        # Environment variable templating logic
        self.config['trainer']['default_root_dir'] = os.path.expandvars(
            self.config['trainer']['default_root_dir'])
        print(f'--trainer.default_root_dir={self.config["trainer"]["default_root_dir"]}')

        # Infer number of nodes
        if 'AWS_BATCH_JOB_NUM_NODES' in os.environ:
            num_nodes = self.config['trainer']['num_nodes']
            batch_num_nodes = int(os.environ['AWS_BATCH_JOB_NUM_NODES'])
            if num_nodes != batch_num_nodes:
                logging.warning(f'--trainer.num_nodes={num_nodes} != '
                                f'$AWS_BATCH_JOB_NUM_NODES={batch_num_nodes}. '
                                f'Setting --trainer.num_nodes={batch_num_nodes}!')
                self.config['trainer']['num_nodes'] = batch_num_nodes
        # Check DeepSpeedPlugin matches trainer; can be removed after
        # https://github.com/PyTorchLightning/pytorch-lightning/pull/7026
        for plugin in self.config['trainer']['plugins']:
            if 'class_path' in plugin and plugin[
                    'class_path'] == 'pytorch_lightning.plugins.training_type.DeepSpeedPlugin':
                if plugin['init_args']['num_nodes'] != self.config['trainer']['num_nodes']:
                    print('Overwriting DeepSpeedPlugin.num_nodes based on trainer.num_nodes.')
                    plugin['init_args']['num_nodes'] = self.config['trainer']['num_nodes']

        # Adjust max_steps
        assert self.config['trainer']['max_steps']
        assert self.config['model']['base_max_steps']
        global_batch_size = self.config['trainer']['num_nodes'] * self.config['trainer'][
            'gpus'] * self.config['data']['mlm_cfg']['batch_size']
        factor = global_batch_size / self.config['model']['optimizer_cfg']['base_batch_size']
        self.config['trainer']['max_steps'] = int(self.config['trainer']['max_steps'] / factor)
        self.config['model']['base_max_steps'] = int(self.config['model']['base_max_steps'] /
                                                     factor)
        print(f'Adjusted trainer.max_steps by {factor} to {self.config["trainer"]["max_steps"]}.')
        print(f'Adjusted base_max_steps by {factor} to {self.config["model"]["base_max_steps"]}.')

        if self.config['export_to_mxnet']:
            return

        # Sanity checks
        if self.config['seed_everything'] is None:
            logging.warning('No seed specified!')
        if self.config['trainer']['precision'] != 16:
            logging.warning('float16 precision recommended')

        assert self.config['trainer']['replace_sampler_ddp']
        assert not self.config['trainer']['min_steps']
        assert not self.config['trainer']['max_epochs']
        assert not self.config['trainer']['min_epochs']

    def before_fit(self):
        if self.config['bf16']:
            # https://github.com/pytorch/pytorch/pull/61002
            th.set_autocast_gpu_dtype(th.bfloat16)

        if self.config['validate_only']:
            self.trainer.validate(model=self.model, datamodule=self.datamodule)
            sys.exit(0)

        if self.config['export_to_mxnet']:
            assert self.trainer.resume_from_checkpoint
            model = self.model.load_from_checkpoint(self.trainer.resume_from_checkpoint)
            from mstar.models.mxnet_1_compat.bert import \
                BERTEncoder as MXBERTEncoder
            from mstar.models.mxnet_1_compat.bert import \
                BERTModel as MXBERTModel
            assert (model.bert_cfg.hidden_dropout_prob == model.bert_cfg.attention_dropout_prob)
            mx_encoder = MXBERTEncoder(
                num_layers=model.bert_cfg.num_layers, units=model.bert_cfg.units,
                hidden_size=model.bert_cfg.hidden_size, max_length=model.bert_cfg.max_length,
                num_heads=model.bert_cfg.num_heads, dropout=model.bert_cfg.hidden_dropout_prob,
                pre_norm=model.bert_cfg.pre_norm, layer_norm_eps=model.bert_cfg.layer_norm_eps)

            mx_model = MXBERTModel(encoder=mx_encoder, vocab_size=model.bert_cfg.vocab_size,
                                   token_type_vocab_size=model.bert_cfg.num_token_types,
                                   units=model.bert_cfg.units, embed_size=model.bert_cfg.units,
                                   embed_dropout=model.bert_cfg.hidden_dropout_prob,
                                   use_pooler=True, use_token_type_embed=True)

            mx_model.initialize()
            # mx_model.hybridize()
            import mxnet as mx
            ones = mx.nd.ones((3, 3))
            valid_length = mx.nd.ones((3, )) * 3
            mx_model(ones, ones, valid_length)

            mx_params = mx_model._collect_params_with_prefix()
            th_params = {k: v for k, v in model.bert.named_parameters()}
            th_params = {
                k.replace('token_type_embed.', 'token_type_embed.0.'): v
                for k, v in th_params.items()
            }
            th_params = {
                k.replace('layer_norm.bias', 'layer_norm.beta'): v
                for k, v in th_params.items()
            }
            th_params = {
                k.replace('layer_norm.weight', 'layer_norm.gamma'): v
                for k, v in th_params.items()
            }
            th_params = {
                k.replace('token_pos_embed.weight', 'encoder.position_weight'): v
                for k, v in th_params.items()
            }
            th_params = {
                k.replace('all_layers.', 'transformer_cells.'): v
                for k, v in th_params.items()
            }
            th_params = {k.replace('attention_proj.', 'proj.'): v for k, v in th_params.items()}
            th_params = {
                k.replace('word_embed.weight', 'word_embed.0.weight'): v
                for k, v in th_params.items()
            }
            th_params = {
                k.replace('embed_layer_norm.', 'encoder.layer_norm.'): v
                for k, v in th_params.items()
            }

            for k in list(th_params.keys()):
                if 'attn_qkv' in k:
                    query, key, value = th.split(th_params[k], th_params[k].shape[0] // 3, dim=0)
                    del th_params[k]
                    th_params[k.replace('attn_qkv.', 'attention_cell.proj_query.')] = query
                    th_params[k.replace('attn_qkv.', 'attention_cell.proj_key.')] = key
                    th_params[k.replace('attn_qkv.', 'attention_cell.proj_value.')] = value

            assert set(mx_params.keys()) == set(th_params.keys())
            for k, v in mx_params.items():
                v.set_data(th_params[k].detach().numpy())

            model.eval()
            input_ids = th.randint(model.bert_cfg.vocab_size, (3, 3))
            assert_allclose(
                model.bert(input_ids, th.ones((3, 3), dtype=th.int),
                           th.ones((3, )) * 3)[0].detach().numpy(),
                mx_model(mx.nd.array(input_ids.cpu().numpy().T), mx.nd.ones((3, 3)),
                         mx.nd.ones((3, )) * 3)[0].transpose((1, 0, 2)).asnumpy(), rtol=1e-4,
                atol=1e-4)

            mx_model.save_parameters('/mnt/mstar.params')
            print('Saved MXNet v1.x parameters to /mnt/mstar.params')
            sys.exit(0)

    def after_fit(self):
        # pytorch_lightning.callbacks.ModelCheckpoint only saves model
        # checkpoint at the end of training if every_n_val_epochs >= 1.
        # Alternative: self.trainer.checkpoint_callbacks[0].dirpath
        self.trainer.save_checkpoint(os.path.join(self.trainer.default_root_dir, 'last.ckpt'),
                                     weights_only=True)


if __name__ == '__main__':
    from jsonargparse import set_config_read_mode
    set_config_read_mode(fsspec_enabled=True)
    multiprocessing.set_start_method('forkserver')
    # th.multiprocessing.set_start_method('forkserver')  # nccl crash
    CLI(MultiTaskBert, MultiTaskDataModule, save_config_callback=None)
