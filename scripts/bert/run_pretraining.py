"""Pretraining on Code"""
import functools
import glob
import hashlib
import logging
import pathlib
import random
import sys
import typing
from dataclasses import dataclass

import mstar
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
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cli import LightningCLI
from torch.nn import functional as F
from torch.utils.data import DataLoader


def qt_collate_fn(indices, *, tbl):
    batch = tbl.take(indices).to_pydict()
    input_id = th.nn.utils.rnn.pad_sequence(
        [th.tensor(ele) for ele in batch['quickthought1'] + batch['quickthought2']],
        batch_first=True)
    segment_id = th.zeros_like(input_id)
    valid_length = th.tensor(batch['validlength1'] + batch['validlength2'])
    mlm_positions = batch['mlmpositions1'] + batch['mlmpositions2']
    # Masked positions with respect to flattened contextual_embedding (batch_size*seq_length,units)
    seq_length = input_id.shape[1]
    mlm_positions = [np.array(pos) + seq_length * i for i, pos in enumerate(mlm_positions)]
    mlm_positions = th.tensor(np.concatenate(mlm_positions).astype(np.int64))
    mlm_labels = batch['mlmlabels1'] + batch['mlmlabels2']
    mlm_labels = th.tensor(np.concatenate(mlm_labels).astype(np.int64))
    return input_id, segment_id, valid_length, mlm_positions, mlm_labels


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
    def __init__(self, qt_dir: str = './qt_feather', ner_dir: str = './ner_feather',
                 distributedsampler_seed: int = 0, batch_size=8, num_workers=4,
                 mmap_folder='/dev/shm/mstar'):
        super().__init__()
        self.qt_files = sorted(
            glob.glob(str(pathlib.Path(qt_dir) / '**' / '*feather'), recursive=True) +
            glob.glob(str(pathlib.Path(qt_dir) / '*feather')))
        self.ner_files = sorted(
            glob.glob(str(pathlib.Path(ner_dir) / '**' / '*feather'), recursive=True) +
            glob.glob(str(pathlib.Path(ner_dir) / '*feather')))
        if not self.qt_files:
            logging.warning('No MLM+QT data!')
        if not self.ner_files:
            logging.warning('No MLM+NER data!')
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mmap_folder = pathlib.Path(mmap_folder)

        self.qt_id = hashlib.shake_256(";".join(str(p)
                                                for p in self.qt_files).encode()).hexdigest(20)
        self.ner_id = hashlib.shake_256(";".join(str(p)
                                                 for p in self.ner_files).encode()).hexdigest(20)

    def prepare_data(self):
        # prepared data may be cached
        if not (self.mmap_folder / self.qt_id / 'meta.pkl').exists():
            (self.mmap_folder / self.qt_id).mkdir(exist_ok=True, parents=True)
            ds = pa.dataset.dataset(self.qt_files, format='feather')
            # Without combining chunks tbl.take is 1000x slower
            tbl = ds.to_table().combine_chunks()
            mstar.utils.shm.serialize(self.mmap_folder / self.qt_id, tbl)
            del tbl  # mmap serialized table instead of keeping in memory

        if not (self.mmap_folder / self.ner_id / 'meta.pkl').exists():
            (self.mmap_folder / self.ner_id).mkdir(exist_ok=True, parents=True)
            ds = pa.dataset.dataset(self.ner_files, format='feather')
            # Without combining chunks tbl.take is 1000x slower
            tbl = ds.to_table().combine_chunks()
            mstar.utils.shm.serialize(self.mmap_folder / self.ner_id, tbl)
            del tbl  # mmap serialized table instead of keeping in memory

    def setup(self, stage: typing.Optional[str] = None):
        # mmap prepared dataset
        self.qt_tbl = mstar.utils.shm.load(self.mmap_folder / self.qt_id)
        self.ner_tbl = mstar.utils.shm.load(self.mmap_folder / self.ner_id)

    def train_dataloader(self):
        # batch_size // 2 for as each sample contains 2 sentences (quickthought)
        qt_loader = DataLoader(np.arange(len(self.qt_tbl)),
                               collate_fn=functools.partial(qt_collate_fn, tbl=self.qt_tbl),
                               batch_size=self.batch_size // 2, num_workers=self.num_workers,
                               pin_memory=True)
        ner_loader = DataLoader(np.arange(len(self.ner_tbl)),
                                collate_fn=functools.partial(ner_collate_fn, tbl=self.ner_tbl),
                                batch_size=self.batch_size, num_workers=self.num_workers,
                                pin_memory=True)
        return {'qt': qt_loader, 'ner': ner_loader}

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError


class CountMetric(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("count", default=th.tensor(0), dist_reduce_fx="sum")

    def update(self, count: th.Tensor):
        self.count += count

    def compute(self):
        return self.count.float()


class LossMetric(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("loss", default=th.tensor(0, dtype=th.float32), dist_reduce_fx="sum")
        self.add_state("count", default=th.tensor(0), dist_reduce_fx="sum")

    def update(self, loss: th.Tensor, count: th.Tensor):
        self.loss += loss
        self.count += count

    def compute(self):
        return self.loss.float() / self.count.float()


@dataclass
class BertModelConfig:
    vocab_size: int = 250027
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
    dtype: str = 'float32'
    layout: str = 'NT'
    compute_layout: str = 'auto'


@dataclass
class OptimizerConfig:
    weight_decay: float = 0.01
    betas: typing.Tuple[float, float] = (0.9, 0.95)
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.01
    phase2: bool = False
    phase1_num_steps: int = 0


@dataclass
class NERConfig:
    vocab_size: float = 81


class MultiTaskBert(pl.LightningModule):
    """MultiTaskBert consists of a BERT backbone and task specific heads.

    Supported tasks must be specified at construction. At forward pass, the
    task name of the supplied batch must be specified.

    """
    def __init__(self, *, bert_cfg: BertModelConfig = BertModelConfig(),
                 optimizer_cfg: OptimizerConfig = OptimizerConfig(),
                 ner_cfg: NERConfig = NERConfig(), single_task_mode: bool = False):
        super().__init__()
        # auto creates self.hparams from the method signature
        self.save_hyperparameters()

        # in lightning the "config" is hparams (for hyperparameters)
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

        self.num_tokens_metric = CountMetric(dist_sync_on_step=True)
        self.mlm_loss_metric = LossMetric(dist_sync_on_step=True)
        self.qt_loss_metric = LossMetric(dist_sync_on_step=True)
        self.mlm_acc_metric = torchmetrics.Accuracy(dist_sync_on_step=True)
        self.qt_acc_metric = torchmetrics.Accuracy(dist_sync_on_step=True)

        if not self.single_task_mode:
            self.ner_decoder = th.nn.Sequential(
                th.nn.Linear(out_features=bert_cfg.units, in_features=bert_cfg.units),
                mstar.layers.get_activation(bert_cfg.activation),
                th.nn.LayerNorm(bert_cfg.units, eps=bert_cfg.layer_norm_eps),
                th.nn.Linear(out_features=ner_cfg.vocab_size, in_features=bert_cfg.units))
            self.ner_loss_metric = LossMetric(dist_sync_on_step=True)
            self.ner_acc_metric = torchmetrics.Accuracy(dist_sync_on_step=True)

        # Initialization
        self.apply(mstar.models.bert.init_weights)

    def configure_optimizers(self):
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
        optimizer = mstar.optimizers.FusedAdam(optim_groups,
                                               lr=self.hparams.optimizer_cfg.learning_rate,
                                               betas=self.hparams.optimizer_cfg.betas)

        # configure the learning rate schedule
        if self.optimizer_cfg.phase2:
            num_steps = self.trainer.max_steps - self.optimizer_cfg.phase1_num_steps
        else:
            num_steps = self.trainer.max_steps
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.optimizer_cfg.warmup_ratio * num_steps,
            num_training_steps=num_steps)

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
        mlm_loss, qt_loss, ner_loss = 0, 0, 0
        # MLM + QT
        if random.random() < 0.8 or self.single_task_mode:
            input_id, segment_id, valid_length, mlm_positions, mlm_labels = batch['qt']
            mlm_features, pooled_out = self.bert(input_id, segment_id, valid_length)
            if self.bert_cfg.layout == 'NT':
                mlm_features = mlm_features.flatten(0, 1)[mlm_positions]
            else:
                mlm_features = th.transpose(mlm_features, 0, 1)\
                                 .flatten(0, 1)[mlm_positions]
            mlm_scores = self.mlm_decoder(mlm_features)
            mlm_loss = F.cross_entropy(mlm_scores, mlm_labels)

            self.mlm_acc_metric(mlm_scores.argmax(dim=1), mlm_labels)
            self.mlm_loss_metric(mlm_loss * len(mlm_labels), len(mlm_labels))

            qt_embeddings = self.quickthought(pooled_out)
            qt_similarity = self._cosine_similarity(qt_embeddings[:len(input_id) // 2],
                                                    qt_embeddings[len(input_id) // 2:])

            qt_label = th.arange(len(input_id) // 2).to(qt_similarity.device)
            qt_loss = F.cross_entropy(qt_similarity, qt_label)

            self.qt_loss_metric(qt_loss * len(input_id), len(input_id))  # batch_size
            self.qt_acc_metric(qt_similarity.argmax(dim=1), qt_label)

            self.log('mlm_loss', self.mlm_loss_metric, on_step=True, on_epoch=False, prog_bar=False,
                     logger=True)
            self.log('mlm_acc', self.mlm_acc_metric, on_step=True, on_epoch=False, prog_bar=False,
                     logger=True)
            self.log('qt_loss', self.qt_loss_metric, on_step=True, on_epoch=False, prog_bar=False,
                     logger=True)
            self.log('qt_acc', self.qt_acc_metric, on_step=True, on_epoch=False, prog_bar=False,
                     logger=True)
        else:  # NER
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
            self.ner_loss_metric(ner_loss * len(ner_labels), len(ner_labels))

            self.log('ner_loss', self.ner_loss_metric, on_step=True, on_epoch=False, prog_bar=False,
                     logger=True)
            self.log('ner_acc', self.ner_acc_metric, on_step=True, on_epoch=False, prog_bar=False,
                     logger=True)

        loss = mlm_loss + qt_loss + ner_loss

        self.num_tokens_metric(valid_length.sum())  # TODO throughput
        self.log('num_tokens', self.num_tokens_metric, on_step=True, on_epoch=False, prog_bar=False,
                 logger=True)

        return loss

    def _cosine_similarity(self, a, b):
        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]
        return th.mm(a_norm, b_norm.transpose(0, 1))


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser) -> None:
        parser.add_argument('--seed', type=int, help='Random seed')
        parser.add_argument('--export_to_mxnet', action="store_true",
                            help='Export trained model for MXNet. Only useful')

    def before_instantiate_classes(self) -> None:
        if self.config['export_to_mxnet']:
            return

        if self.config['seed'] is not None:
            pl.seed_everything(self.config['seed'])

        # Sanity checks
        if self.config['seed'] is None:
            logging.warning('No seed specified!')
        if self.config['trainer']['precision'] != 16:
            logging.warning('float16 precision recommended')

        assert self.config['trainer']['max_steps']
        assert self.config['trainer']['replace_sampler_ddp']
        assert not self.config['trainer']['min_steps']
        assert not self.config['trainer']['max_epochs']
        assert not self.config['trainer']['min_epochs']

    def before_fit(self):
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
                pre_norm=model.bert_cfg.pre_norm)

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
            assert_allclose(
                model.bert(th.ones((3, 3), dtype=th.int), th.ones((3, 3), dtype=th.int),
                           th.ones((3, )) * 3)[0].detach().numpy(),
                mx_model(mx.nd.ones((3, 3)), mx.nd.ones((3, 3)),
                         mx.nd.ones((3, )) * 3)[0].transpose((1, 0, 2)).asnumpy(), rtol=1e-05,
                atol=1e-05)

            mx_model.save_parameters('/mnt/mstar.params')
            print('Saved MXNet v1.x parameters to /mnt/mstar.params')
            sys.exit(0)

    def after_fit(self):
        self.trainer.save_checkpoint("/mnt/mstar.ckpt")


if __name__ == '__main__':
    CLI(MultiTaskBert, MultiTaskDataModule, save_config_callback=None,
        trainer_defaults={'callbacks': [ModelCheckpoint(every_n_train_steps=10000)]})
