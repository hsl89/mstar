"""Pretraining on Code"""
import argparse
import functools
import hashlib
import json
import logging
import os
import pathlib
import random
import shutil
import time
import warnings
from contextlib import suppress

import mstar
import numpy as np
import pyarrow as pa
import pyarrow.compute
import pyarrow.dataset
import torch as th
import transformers
from fairscale.nn.data_parallel import ShardedDataParallel
from fairscale.optim.grad_scaler import ShardedGradScaler
from fairscale.optim.oss import OSS
from torch import distributed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter


def repeat(iterable, count=None, *, set_epoch=False):
    if count is None:
        i = 0
        while True:
            if set_epoch:
                iterable.sampler.set_epoch(i)
            for sample in iterable:
                yield sample
            i += 1
    else:
        for i in range(count):
            if set_epoch:
                iterable.sampler.set_epoch(i)
            for sample in iterable:
                yield sample


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description=__doc__)
    # Input / output
    group = parser.add_argument_group('Input / output')
    group.add_argument('--input-files', type=pathlib.Path, nargs='+')
    group.add_argument(
        '--mmap-folder', type=pathlib.Path, default='/dev/shm/mstar',
        help='Folder to place mmap files for sharing dataset accross local worker processes.')
    group.add_argument('--lang', type=str, choices=['python'])
    group.add_argument('--ckpt_dir', type=str, default='./ckpt_dir',
                       help='Path to checkpoint directory')
    group.add_argument('--log_interval', type=int, default=50, help='Report interval')
    group.add_argument('--ckpt_interval', type=int, default=25000, help='Checkpoint interval')
    group.add_argument('--verbose', action='store_true', help='verbose logging')

    # model
    group = parser.add_argument_group('Model')
    group.add_argument('--model_name', type=str, default='google_en_uncased_bert_base',
                       choices=mstar.models.bert.bert_cfg_reg.list_keys(),
                       help='Name of the model configuration.')

    # training
    group = parser.add_argument_group('Training')
    group.add_argument('--seed', type=int, default=100, help='Random seed')
    group.add_argument('--batch_size', type=int, default=8,
                       help='Batch size per GPU in a minibatch.')
    group.add_argument(
        '--num_accumulated', type=int, default=1,
        help='Number of batches for gradient accumulation. '
        'total_batch_size = batch_size_per_worker * num_worker * accumulate.')
    group.add_argument('--num_steps', type=int, default=20, help='Number of optimization steps')
    group.add_argument('--start_step', type=int, default=0,
                       help='Start optimization step from the checkpoint.')
    group.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    group.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
    group.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm.')
    group.add_argument('--warmup_ratio', type=float, default=0.01,
                       help='Ratio of warmup steps in the learning rate scheduler.')
    group.add_argument('--const_ratio', type=float, default=0.0,
                       help='Ratio of constant steps in the learning rate scheduler.')

    group.add_argument('--num_dataloader_workers', type=int, default=4,
                       help='Number of workers to pre-process dataset.')

    # phase 2
    parser.add_argument('--phase2', action='store_true', help='phase 2 training')
    parser.add_argument('--phase1_num_steps', type=int, help='number of steps for phase 1')

    # computation and communication
    group = parser.add_argument_group('Computation and communication')
    group.add_argument("--local_rank", type=int, default=-1, help="Rank in distributed training")
    group.add_argument("--fp16", action=mstar.utils.misc.BooleanOptionalAction, default=True,
                       help="Whether to use 16-bit (mixed) precision instead of 32-bit.")
    parser.add_argument(
        "--ZeRO", action=mstar.utils.misc.BooleanOptionalAction, default=False,
        help="Use ZeRO parameter and optimizer state sharding. "
        "Helps speed-up and reduce memory usage of large models.")
    group.add_argument("--cuda", action=mstar.utils.misc.BooleanOptionalAction, default=True,
                       help="Use Cuda if available.")
    group.add_argument("--activation-checkpointing", action=mstar.utils.misc.BooleanOptionalAction,
                       default=False, help="Trade compute for memory by checkpointing activations.")
    args = parser.parse_args()

    # Yet to be supported settings
    assert not args.activation_checkpointing  # TODO

    return args


def final_save(model, save_dir, cfg):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'model.yml'), 'w') as f:
        f.write(cfg.dump())
    th.save(model.state_dict(), os.path.join(save_dir, 'model.params'))
    logging.info('Statistics:')
    old_names = os.listdir(save_dir)
    for old_name in old_names:
        new_name, long_hash = mstar.utils.misc.naming_convention(save_dir, old_name)
        old_path = os.path.join(save_dir, old_name)
        new_path = os.path.join(save_dir, new_name)
        shutil.move(old_path, new_path)
        file_size = os.path.getsize(new_path)
        logging.info('\t{}/{} {} {}'.format(save_dir, new_name, long_hash, file_size))


def parameters_option(step_num, model, args, option='Saving', ctx_l=None):
    """Save or load the model parameter, marked by step_num."""
    param_path = os.path.join(args.ckpt_dir, f'{step_num:07}.params')
    logging.info(f'[Step {step_num}], {option} model params to/from {param_path}.')
    if option == 'Saving':
        try:  # Handle "module." prefix https://github.com/pytorch/pytorch/issues/9176
            state_dict = model.module.state_dict()
        except AttributeError:
            state_dict = model.state_dict()
        th.save(state_dict, param_path)
    elif option == 'Loading':
        model.load_state_dict(th.load(param_path, map_location=args.device))
    else:
        raise NotImplementedError('Unknown Option: {}'.format(option))


def states_option(step_num, optimizer, args, option='Saving'):
    """Save or load the trainer states, marked by step_num and local rank."""
    state_path = os.path.join(args.ckpt_dir, f'{step_num:07}.states.{args.local_rank:02}')
    logging.info(f'[Step {step_num}], {option} trainer states to/from {state_path}.')
    if option == 'Saving':
        th.save(optimizer.state_dict(), state_path)
    elif option == 'Loading':
        optimizer.load_state_dict(th.load(state_path))
    else:
        raise NotImplementedError('Unknown Option: {}'.format(option))


def set_seed(seed):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True


def get_world_size(args):
    if args.local_rank != -1:
        return distributed.get_world_size()
    return 1


def collate_fn(indices, *, args, tbl):
    batch = tbl.take(indices).to_pydict()
    input_id = th.nn.utils.rnn.pad_sequence(
        [th.tensor(ele) for ele in batch['quickthought1'] + batch['quickthought2']],
        batch_first=True)
    segment_id = th.zeros_like(input_id)
    valid_length = th.tensor(batch['validlength1'] + batch['validlength2'])
    mlm_positions = batch['mlmpositions1'] + batch['mlmpositions2']
    # Masked positions with respect to flattened contextual_embedding (batch_size * seq_length, units)
    seq_length = input_id.shape[1]
    mlm_positions = [np.array(pos) + seq_length * i for i, pos in enumerate(mlm_positions)]
    mlm_positions = th.tensor(np.concatenate(mlm_positions).astype(np.int64))
    mlm_labels = batch['mlmlabels1'] + batch['mlmlabels2']
    mlm_labels = th.tensor(np.concatenate(mlm_labels).astype(np.int64))
    return input_id, segment_id, valid_length, mlm_positions, mlm_labels


def train(args, *, tbl):
    cfg = mstar.models.bert.google_en_uncased_bert_base()
    model = mstar.models.bert.QTBertForPretrain(cfg)
    model.to(args.device)

    if args.start_step:
        logging.info('Restart training from {}'.format(args.start_step))
        parameters_option(args.start_step, model, args, 'Loading')
    else:
        model.apply(mstar.models.bert.init_weights)

    writer = None
    if args.local_rank in (-1, 0):
        writer = SummaryWriter(log_dir=os.path.join(args.ckpt_dir, 'tensorboard'))

    # pin_memory=False due to lack of https://github.com/pytorch/pytorch/commit/54ce171f16c8859f829dde09f87c364c8a6b4130
    sampler = RandomSampler(tbl) if args.local_rank == -1 else DistributedSampler(
        tbl, seed=args.seed)
    # batch_size // 2 for QuickThought
    train_dataloader = DataLoader(np.arange(len(tbl)), sampler=sampler,
                                  collate_fn=functools.partial(collate_fn, args=args, tbl=tbl),
                                  batch_size=args.batch_size // 2,
                                  num_workers=args.num_dataloader_workers, pin_memory=True)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay':
        args.weight_decay
    }, {
        'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]

    optimizer_arguments = {"lr": args.lr}
    if get_world_size(args) > 1 and args.ZeRO:
        optimizer = OSS(params=model.parameters(), optim=mstar.optimizers.FusedLANS,
                        **optimizer_arguments)
        model = ShardedDataParallel(model, optimizer)
    elif get_world_size(args) > 1:
        optimizer = mstar.optimizers.FusedLANS(optimizer_grouped_parameters, **optimizer_arguments)
        model = DistributedDataParallel(model, device_ids=[args.local_rank],
                                        output_device=args.local_rank, find_unused_parameters=True)
    else:
        optimizer = mstar.optimizers.FusedLANS(optimizer_grouped_parameters, **optimizer_arguments)

    save_interval = args.ckpt_interval
    logging.info(f'#Total Training Steps={args.num_steps}, '
                 f'Warmup Steps={args.warmup_ratio * args.num_steps}, '
                 f'Save Interval={save_interval}')
    assert not args.const_ratio
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_ratio * args.num_steps,
        num_training_steps=args.num_steps)
    # scheduler = mstar.optimizers.schedules.get_warmup_linear_const_decay_poly_schedule(
    #     optimizer, total_steps=args.num_steps, warmup_ratio=args.warmup_ratio,
    #     const_ratio=args.const_ratio)

    if args.start_step:
        logging.info(f'Restart training from {args.start_step}')
        states_option(args.start_step, optimizer, args, 'Loading')

    ce_loss_fn = th.nn.CrossEntropyLoss()
    step_num = args.start_step
    if args.phase2:
        step_num -= args.phase1_num_steps
    running_num_tks, running_grad_norm = 0, 0
    running_mlm_loss, running_qt_loss, running_mlm_acc, running_qt_acc = 0, 0, 0, 0

    train_start_time = time.time()
    tic = time.time()
    model.zero_grad()
    if get_world_size(args) > 1 and args.ZeRO:
        scaler = ShardedGradScaler() if args.fp16 else None
    else:
        scaler = th.cuda.amp.GradScaler() if args.fp16 else None

    train_iter = repeat(train_dataloader, set_epoch=args.local_rank != -1)
    while step_num < args.num_steps:
        step_num += 1
        for accum_step in range(args.num_accumulated):
            (input_id, segment_id, valid_length, mlm_positions, mlm_labels) = next(train_iter)
            (input_id, segment_id, valid_length, mlm_positions,
             mlm_labels) = (arr.to(args.device) for arr in next(train_iter))

            model.train()
            accumulation = ((accum_step + 1) % args.num_accumulated != 0)
            with model.no_sync() if get_world_size(args) > 1 and accumulation else suppress():
                with th.cuda.amp.autocast(enabled=args.fp16):
                    _, pooled_out, mlm_scores, qt_similarity = model(input_id, segment_id,
                                                                     valid_length, mlm_positions)
                    mlm_loss = ce_loss_fn(mlm_scores, mlm_labels)
                    qt_label = th.arange(len(input_id) // 2, device=args.device)
                    qt_loss = ce_loss_fn(qt_similarity, qt_label)
                    loss = mlm_loss + qt_loss
                if args.num_accumulated > 1:
                    loss = loss / args.num_accumulated
                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                with th.no_grad():
                    qt_acc = (qt_similarity.argmax(dim=1) == qt_label).sum() / (len(input_id) // 2)
                    mlm_acc = (mlm_scores.argmax(dim=1) == mlm_labels).sum() / len(mlm_labels)

            # Gather information from all workers for accurate statistics
            reduced_num_tokens = valid_length.sum()
            if get_world_size(args) > 1:
                distributed.all_reduce(reduced_num_tokens)
            reduced_num_mlm_tokens = th.tensor(len(mlm_labels), device=args.device)
            if get_world_size(args) > 1:
                distributed.all_reduce(reduced_num_mlm_tokens)
            reduced_loss_mlm = mlm_loss.detach().clone() * len(mlm_labels) / reduced_num_mlm_tokens
            if get_world_size(args) > 1:
                distributed.all_reduce(reduced_loss_mlm)
            reduced_acc_mlm = mlm_acc.detach().clone() * len(mlm_labels) / reduced_num_mlm_tokens
            if get_world_size(args) > 1:
                distributed.all_reduce(reduced_acc_mlm)
            reduced_bs = th.tensor(len(input_id), device=args.device)
            if get_world_size(args) > 1:
                distributed.all_reduce(reduced_bs)
            reduced_loss_qt = qt_loss.detach().clone() * len(input_id) / reduced_bs
            if get_world_size(args) > 1:
                distributed.all_reduce(reduced_loss_qt)
            reduced_acc_qt = qt_acc.detach().clone() * len(input_id) / reduced_bs
            if get_world_size(args) > 1:
                distributed.all_reduce(reduced_acc_qt)

            running_num_tks += reduced_num_tokens.item()
            running_mlm_loss += reduced_loss_mlm.item()
            running_mlm_acc += reduced_acc_mlm.item()
            running_qt_loss += reduced_loss_qt.item()
            running_qt_acc += reduced_acc_qt.item()

            if not accumulation:
                if args.fp16:
                    scaler.unscale_(optimizer)  # unscale for gradient clipping
                if get_world_size(args) > 1 and args.ZeRO:
                    total_norm = optimizer.clip_grad_norm(args.max_grad_norm)
                else:
                    total_norm = th.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    if get_world_size(args) > 1:
                        distributed.all_reduce(total_norm)
                        total_norm /= get_world_size(args)
                running_grad_norm += total_norm

                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                with warnings.catch_warnings():
                    # Scheduler may warn if optimizer.step() call is skipped
                    # due to invalid gradients detected by scaler.
                    warnings.simplefilter("ignore", UserWarning)
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        if step_num % args.log_interval == 0:
            toc = time.time()
            wps = running_num_tks / (toc - tic)
            eta = (args.num_steps - step_num) / (step_num / (toc - train_start_time)) / 3600
            interval = args.log_interval * args.num_accumulated
            logging.info(f'[Step {step_num}], LR={scheduler.get_last_lr()[0]:.6f}, '
                         f'Loss MLM/QT={running_mlm_loss / interval:.4f}/'
                         f'{running_qt_loss / interval:.4f}, '
                         f'Acc MLM/QT={running_mlm_acc / interval:.4f}/'
                         f'{running_qt_acc / interval:.4f}, '
                         f'Grad_norm={running_grad_norm / interval:.4f}, '
                         f'Time cost={toc - tic:.2f}, '
                         f'Throughput={wps:.2f} tokens/s, ETA={eta:.2f}h')
            if args.local_rank in (-1, 0):
                save_step_num = step_num if not args.phase2 else step_num + args.phase1_num_steps
                writer.add_scalar('Throughput_wps', wps, save_step_num)
                writer.add_scalar('Loss/MLM', running_mlm_loss / interval, save_step_num)
                writer.add_scalar('Loss/QT', running_qt_loss / interval, save_step_num)
                writer.add_scalar('Acc/MLM', running_mlm_acc / interval, save_step_num)
                writer.add_scalar('Acc/QT', running_qt_acc / interval, save_step_num)
                writer.add_scalar('LR', scheduler.get_last_lr()[0], save_step_num)
                writer.add_scalar('Grad_norm', running_grad_norm / interval, save_step_num)
            running_num_tks, running_grad_norm = 0, 0
            running_mlm_loss, running_qt_loss, running_mlm_acc, running_qt_acc = 0, 0, 0, 0
            tic = time.time()

        # Saving
        if step_num % save_interval == 0 or step_num >= args.num_steps:
            save_step_num = step_num if not args.phase2 else step_num + args.phase1_num_steps
            states_option(save_step_num, optimizer, args, 'Saving')
            if args.local_rank in (0, -1):
                parameters_option(save_step_num, model, args, 'Saving')

    logging.info('Finish training step: %d', step_num)
    train_end_time = time.time()
    logging.info('Train cost={:.1f} s'.format(train_end_time - train_start_time))

    if args.local_rank in (0, -1):
        save_dir = os.path.join(args.ckpt_dir, args.model_name)
        final_save(model, save_dir, cfg)


def main():
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    mstar.utils.misc.logging_config(args.ckpt_dir, name='pretrain_bert_' + str(args.local_rank),
                                    level=level, console=(args.local_rank in (0, -1)))
    # Setup CUDA, GPU & distributed training
    local_size = 1
    if th.cuda.is_available() and args.cuda:
        th.cuda.set_device(args.local_rank if args.local_rank != -1 else 0)
        args.device = th.device("cuda", args.local_rank if args.local_rank != -1 else 0)
        if args.local_rank != -1:
            distributed.init_process_group(backend='nccl')
            local_size = th.cuda.device_count()
    else:
        args.device = th.device("cpu")
        if args.local_rank != -1:
            distributed.init_process_group(backend='gloo')

    logging.info(args)
    logging.debug('Random seed set to {}'.format(args.seed))
    set_seed(args.seed)
    logging.info(f'Training info: num_workers: {get_world_size(args)}, '
                 f'local rank: {args.local_rank}')

    train_tbl_id = hashlib.shake_256( \
        ";".join(str(p) for p in args.input_files).encode()).hexdigest(20)
    if args.local_rank not in (-1, 0):
        distributed.barrier()  # Wait for dataset
        train_tbl = mstar.utils.shm.load(args.mmap_folder / train_tbl_id)
    else:  # Main process
        if (args.mmap_folder / train_tbl_id / 'meta.pkl').exists():
            if args.local_rank != -1:
                distributed.barrier()  # Indicate dataset is ready
            train_tbl = mstar.utils.shm.load(args.mmap_folder / train_tbl_id)
        else:
            (args.mmap_folder / train_tbl_id).mkdir(exist_ok=True, parents=True)
            ds = pa.dataset.dataset(args.input_files, format='feather')
            # Without combining chunks tbl.take is 1000x slower
            train_tbl = ds.to_table().combine_chunks()
            mstar.utils.shm.serialize(args.mmap_folder / train_tbl_id, train_tbl)
            if args.local_rank != -1:
                distributed.barrier()  # Indicate dataset is ready
            del train_tbl
            train_tbl = mstar.utils.shm.load(args.mmap_folder / train_tbl_id)

    step_size = args.batch_size * args.num_accumulated * get_world_size(args)
    logging.info(f'Dataset has {len(train_tbl)} rows.')
    logging.info(f'Sampling {step_size} rows per step ({step_size/len(train_tbl)*100:.2f}% data)')
    logging.info(f'Will iterate over the dataset during {args.num_steps} training steps '
                 f'{args.num_steps * step_size/len(train_tbl):.2f} times.')

    train(args, tbl=train_tbl)


if __name__ == '__main__':
    main()
