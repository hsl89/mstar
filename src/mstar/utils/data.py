from dataclasses import dataclass
import functools
import itertools
import math
import os
import random
import json
from smart_open import open as smart_open
from concurrent.futures import ProcessPoolExecutor

import torch as th
from transformers import AutoTokenizer

import mstar


@dataclass
class MLMConfig:
    max_seq_length: int = 128
    mlm_probability: float = 0.15
    short_seq_probability: float = 0.1
    batch_size: int = 8
    subsample: bool = False
    use_nsp: bool = False
    file_type: str = 'txt'


@dataclass
class S3DataConfig:
    bucket: str = 'mstar-data'
    prefix: str = 'wiki-20210401-processed-resampled'
    pattern: str = '*/part*.txt.gz'
    eval_splits: int = 1
    tokenizer: str = 'bert-base-multilingual-cased'


@dataclass
class MLMS3DataConfig(MLMConfig, S3DataConfig):
    pass


def mlm_mask(labels, mlm_probability, num_tokens,  # pylint: disable=too-many-arguments
             mask_idx, cls_idx, sep_idx):
    """Generate masked language model masks.

    Parameters
    ----------
    labels : torch.Tensor
        Unmasked input array.
    mlm_probability : float
        Probability for masking tokens in input array.
    num_tokens : int
        Vocabulary size.
    mask_idx : int
        Index for [MASK] token.
    cls_idx : int
        Index for [CLS] token.
    sep_idx : int
        Index for [SEP] token.
    """

    inputs = labels.clone()

    probability_matrix = th.full(inputs.shape, mlm_probability)
    special_tokens_mask = (inputs == cls_idx) | (inputs == sep_idx)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    masked_indices = th.bernoulli(probability_matrix).bool()
    indices_replaced = th.bernoulli(th.full(inputs.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = mask_idx
    # 10% of the time, we replace masked input tokens with random word
    indices_random = (th.bernoulli(th.full(inputs.shape, 0.5)).bool() & masked_indices
                      & ~indices_replaced)
    random_words = th.randint(num_tokens, inputs.shape, dtype=th.long)
    inputs[indices_random] = random_words[indices_random]
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged

    return inputs, masked_indices


class MLMFileTokenizer:
    # pylint: disable=too-many-locals, too-many-statements, too-many-nested-blocks
    def __init__(self, cfg, tokenizer):
        """Tokenizes a file and generate MLM masking.

        Parameters
        ----------
        cfg : MLMConfig
            Configuration for masked language modeling.
        tokenizer : PretrainedTokenizer
            Huggingface pretrained tokenizer.
        """
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer

    def __call__(self, path):  # noqa: MC0001
        cfg, tokenizer = self.cfg, self.tokenizer
        cls_idx, sep_idx, mask_idx = tokenizer.convert_tokens_to_ids(
            [tokenizer.cls_token, tokenizer.sep_token, tokenizer.mask_token])

        assert cfg.file_type in {'txt', 'jsonl'}
        with smart_open(path, 'r') as f:
            if cfg.file_type == 'txt':
                all_documents = [
                    [y for y in x.split() if y]
                    for x in f.read().split('\n\n') if x
                ]
            elif cfg.file_type == 'jsonl':
                all_documents = [
                    [y for y in json.loads(line)['text'].split() if y]
                    for line in f.readlines() if line
                ]
            if cfg.subsample:
                all_documents = random.sample(all_documents, math.ceil(0.05 * len(all_documents)))
            num_sentences = [len(p) for p in all_documents]
            sentences = []
            for doc in all_documents:
                sentences.extend(doc)
            sentences = [  # Tokenize
                line.ids for line in tokenizer._tokenizer.encode_batch(
                    sentences, add_special_tokens=False, is_pretokenized=False)
            ]
            all_documents = []
            cnt = 0
            for num_sent in num_sentences:
                all_documents.append([s for s in sentences[cnt:cnt + num_sent] if s])
                cnt += num_sent

            toks, seg_ids, val_lengths = [], [], []
            if cfg.use_nsp:
                nsp_labels = []

            for document_index, document in enumerate(all_documents):
                # Account for [CLS], [SEP], [SEP]
                max_num_tokens = cfg.max_seq_length - 3

                # We *usually* want to fill up the entire sequence since we are padding
                # to `max_seq_length` anyways, so short sequences are generally wasted
                # computation. However, we *sometimes*
                # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
                # sequences to minimize the mismatch between pre-training and fine-tuning.
                # The `target_seq_length` is just a rough target however, whereas
                # `max_seq_length` is a hard limit.
                target_seq_length = max_num_tokens
                if random.random() < cfg.short_seq_probability:
                    target_seq_length = random.randint(2, max_num_tokens)

                # We DON'T just concatenate all of the tokens from a document into a long
                # sequence and choose an arbitrary split point because this would make the
                # next sentence prediction task too easy. Instead, we split the input into
                # segments "A" and "B" based on the actual "sentences" provided by the user
                # input.
                current_chunk = []
                current_length = 0
                i = 0
                while i < len(document):
                    segment = document[i]
                    current_chunk.append(segment)
                    current_length += len(segment)
                    if i == len(document) - 1 or current_length >= target_seq_length:
                        if current_chunk:
                            # `a_end` is how many segments from `current_chunk` go into the `A`
                            # (first) sentence.
                            a_end = 1
                            if len(current_chunk) >= 2:
                                a_end = random.randint(1, len(current_chunk) - 1)

                            tokens_a = []
                            for j in range(a_end):
                                tokens_a.extend(current_chunk[j])

                            tokens_b = []

                            # Random next
                            is_random_next = False
                            if len(current_chunk) == 1 or random.random() < 0.5:
                                is_random_next = True
                                target_b_length = target_seq_length - len(tokens_a)

                                # This should rarely go for more than one iteration for large
                                # corpora. However, just to be careful, we try to make sure that
                                # the random document is not the same as the document
                                # we're processing.
                                for _ in range(10):
                                    random_document_index = random.randint(
                                        0,
                                        len(all_documents) - 1)
                                    if random_document_index != document_index:
                                        break

                                random_document = all_documents[random_document_index]
                                random_start = random.randint(0, len(random_document) - 1)
                                for j in range(random_start, len(random_document)):
                                    tokens_b.extend(random_document[j])
                                    if len(tokens_b) >= target_b_length:
                                        break
                                # We didn't actually use these segments so we "put them back" so
                                # they don't go to waste.
                                num_unused_segments = len(current_chunk) - a_end
                                i -= num_unused_segments

                            # Actual next
                            else:
                                is_random_next = False
                                for j in range(a_end, len(current_chunk)):
                                    tokens_b.extend(current_chunk[j])

                            # truncate_seq_pair
                            while True:
                                total_length = len(tokens_a) + len(tokens_b)
                                if total_length <= max_num_tokens:
                                    break
                                trunc_tokens = tokens_a if len(tokens_a) > len(
                                    tokens_b) else tokens_b
                                assert len(trunc_tokens) >= 1
                                # We want to sometimes truncate from the front and sometimes
                                # from the back to add more randomness and avoid biases.
                                if random.random() < 0.5:
                                    del trunc_tokens[0]
                                else:
                                    trunc_tokens.pop()

                            assert len(tokens_a) >= 1
                            assert len(tokens_b) >= 1

                            tokens = []
                            segment_ids = []
                            tokens.append(cls_idx)
                            segment_ids.append(0)
                            for token in tokens_a:
                                tokens.append(token)
                                segment_ids.append(0)

                            tokens.append(sep_idx)
                            segment_ids.append(0)

                            for token in tokens_b:
                                tokens.append(token)
                                segment_ids.append(1)
                            tokens.append(sep_idx)
                            segment_ids.append(1)

                            # Pad
                            valid_length = len(tokens)
                            if len(tokens) < cfg.max_seq_length:
                                tokens.extend([0] * (cfg.max_seq_length - valid_length))
                                segment_ids.extend([0] * (cfg.max_seq_length - valid_length))

                            toks.append(tokens)
                            seg_ids.append(segment_ids)
                            val_lengths.append(valid_length)
                            if cfg.use_nsp:
                                nsp_labels.append(is_random_next)
                        current_chunk = []
                        current_length = 0
                    i += 1

            labels = th.as_tensor(toks)
            segment_ids = th.as_tensor(seg_ids)
            valid_length = th.as_tensor(val_lengths)

            inputs, masked_indices = mlm_mask(
                labels, cfg.mlm_probability, len(tokenizer),
                mask_idx, cls_idx, sep_idx
            )

            if cfg.use_nsp:
                nsp_labels = th.as_tensor(nsp_labels).type(th.long)
                return inputs, labels, masked_indices, segment_ids, valid_length, nsp_labels
            return inputs, labels, masked_indices, segment_ids, valid_length


def get_train_and_eval_paths(s3_cfg):
    """Return file paths for train/eval according to the data config.

    Parameters
    ----------
    s3_cfg : S3DataConfig
        Configuration for files.

    Returns
    -------
    train_paths, eval_paths
    """
    # TODO(szha): generalize file matching to make the class fs-generic.
    matched_keys = mstar.utils.misc.list_matched_s3_objects(
        s3_cfg.bucket, s3_cfg.prefix, s3_cfg.pattern)
    assert 0 <= s3_cfg.eval_splits <= len(matched_keys)
    train_keys, eval_keys = len(matched_keys) - s3_cfg.eval_splits, s3_cfg.eval_splits
    paths = [f's3://{s3_cfg.bucket}/{shard}' for shard in matched_keys]
    return paths[0:train_keys], paths[-eval_keys:]


def batch_iter_from_shard(shard, batch_size, shuffle, keep_last_batch, use_nsp):
    # pylint: disable=too-many-locals
    if use_nsp:
        inputs, labels, masked_indices, segment_ids, valid_length, nsp_labels = shard
    else:
        inputs, labels, masked_indices, segment_ids, valid_length = shard
    assert len(inputs.shape) == 2
    if shuffle:
        indices = th.randperm(inputs.shape[0])
    else:
        indices = th.arange(inputs.shape[0])
    if keep_last_batch:
        num_batches = (inputs.shape[0] + batch_size - 1) // batch_size
    else:
        num_batches = inputs.shape[0] // batch_size
    for i in range(num_batches):
        batch_indices = indices[i * batch_size:(i + 1) * batch_size]
        batch_input_id = inputs[batch_indices]
        batch_segment_id = segment_ids[batch_indices]
        batch_valid_length = valid_length[batch_indices]
        batch_mlm_positions = th.nonzero(masked_indices[batch_indices].flatten()).flatten()
        batch_mlm_labels = labels[batch_indices][masked_indices[batch_indices]]
        if use_nsp:
            batch_nsp_labels = nsp_labels[batch_indices]
            yield (batch_input_id, batch_segment_id, batch_valid_length, batch_mlm_positions,
                   batch_mlm_labels, batch_nsp_labels)
        else:
            yield (batch_input_id, batch_segment_id, batch_valid_length, batch_mlm_positions,
                   batch_mlm_labels)


class DistributedMLM(th.utils.data.IterableDataset):
    # pylint: disable=too-many-arguments, too-many-locals
    def __init__(self, paths, mlm_cfg, infinite=True, eager=False, shuffle=True,
                 keep_last_batch=False):
        """Masked language modeling data from a set of text files.

        If torch distributed is initialized, the set of files is partitioned
        among the ranks (unless validation=True). Thus, the number of files
        must be greater or equal than the number of ranks. Validation files
        specified via eval_splits are excluded.

        When this dataset is used with multiple workers, this class will perform
        automatic sharding. If the dataset is initialized with eager mode, all
        workers will load all shards and then split the data across workers at sample
        level. If the dataset is initialized with lazy mode, the number of shards each
        worker has will be::

            lcm(#worker, #shards) / #worker

        The dataset guarantees that in this setting, if all workers exhaust all shards,
        each shard will appear the same number of times, and no two workers will have the
        same shard at any time.

        Parameters
        ----------
        paths : list of str
            List of file paths to read.
        mlm_cfg : MLMS3DataConfig
            Dataset configuration.
        infinite : bool, default True
            If True, DistributedMLM is an infinite iterable on a loop based on
            the local shards. This is useful to prevent gaps in multi-processing
            based prefetching. Processed files are not cached and each re-processing
            will use a separate random mask.
        eager : bool, default False
            If True, DistributedMLM eagerly processes all shards once on every
            worker, and then partitions tokenized samples across ranks.
        shuffle : bool, default True
            If True, DistributedMLM shuffles the order of shards.
        keep_last_batch : bool, default False
            Whether to keep the last batch whose size is smaller than configured
            batch size.
        """
        super().__init__()
        assert not (infinite and eager), \
            'Dataset loading cannot be both infinite and eager.'
        self._num_cpus = os.cpu_count()
        self._world_size = 1
        global_rank = 0

        worker_info = th.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        if th.distributed.is_initialized():
            self._num_cpus = math.ceil(os.cpu_count() / th.cuda.device_count())
            global_rank = th.distributed.get_rank()
            self._world_size = th.distributed.get_world_size()
            assert self._world_size <= len(paths)
        self._paths = paths
        self._mlm_cfg = mlm_cfg

        self._tokenizer = AutoTokenizer.from_pretrained(mlm_cfg.tokenizer)
        self._cls_idx, self._sep_idx, self._mask_idx = self._tokenizer.convert_tokens_to_ids(
            [self._tokenizer.cls_token, self._tokenizer.sep_token, self._tokenizer.mask_token])
        self._num_tokens = len(self._tokenizer)
        self._infinite = infinite
        self._eager = eager
        self._shuffle = shuffle
        self._keep_last_batch = keep_last_batch

        if eager:
            toks = None
            self._shard_paths = self._paths
            # Call hf/tokenizers in a separate process, to avoid "The current
            # process just got forked, after parallelism has already been used.
            # Disabling parallelism to avoid deadlocks..."
            ex = ProcessPoolExecutor(max_workers=1)
            if global_rank == 0:
                tokenized = list(ex.map(MLMFileTokenizer(mlm_cfg, self._tokenizer),
                                        self._shard_paths))
                fields = zip(*tokenized)
                fields = [th.cat(f) for i, f in enumerate(fields) if i in {1, 3, 4, 5}]
                toks, self._segment_ids, self._valid_lengths = fields[0], fields[1], fields[2]
                objects = [toks, self._segment_ids, self._valid_lengths]
                if mlm_cfg.use_nsp:
                    self._nsp_labels = fields[3]
                    objects.append(self._nsp_labels)
                if self._world_size > 1:
                    th.distributed.broadcast_object_list(objects, src=0)
            else:
                if mlm_cfg.use_nsp:
                    objects = [None, None, None, None]
                    th.distributed.broadcast_object_list(objects, src=0)
                    toks, self._segment_ids, self._valid_lengths, self._nsp_labels = objects
                else:
                    objects = [None, None, None]
                    th.distributed.broadcast_object_list(objects, src=0)
                    toks, self._segment_ids, self._valid_lengths = objects

            start = global_rank * num_workers + worker_id
            end = (len(toks) // (self._world_size * num_workers)) * self._world_size * num_workers
            step = self._world_size * num_workers
            self._labels = toks[start:end:step]
            self._segment_ids = self._segment_ids[start:end:step]
            self._valid_length = self._valid_lengths[start:end:step]
            if mlm_cfg.use_nsp:
                self._nsp_labels = self._nsp_labels[start:end:step]
        else:
            shard_ids = mstar.utils.misc.generate_shards(
                len(self._paths), num_workers)[worker_id]
            self._shard_paths = [self._paths[i] for i in shard_ids]

    @property
    def shard_paths(self):
        return self._shard_paths

    def _eager_iter(self):
        cfg = self._mlm_cfg

        labels = self._labels.clone()
        segment_ids = self._segment_ids.clone()
        valid_length = self._valid_lengths.clone()
        if cfg.use_nsp:
            nsp_labels = self._nsp_labels.clone()

        inputs, masked_indices = mlm_mask(
            labels, cfg.mlm_probability, len(self._tokenizer),
            self._mask_idx, self._cls_idx, self._sep_idx
        )

        if cfg.use_nsp:
            shard = (inputs, labels, masked_indices, segment_ids, valid_length, nsp_labels)
        else:
            shard = (inputs, labels, masked_indices, segment_ids, valid_length)
        yield from batch_iter_from_shard(
            shard, cfg.batch_size, self._shuffle, self._keep_last_batch, cfg.use_nsp
        )

    def _lazy_iter(self):
        cfg = self._mlm_cfg

        paths = self.shard_paths
        if self._infinite:
            # Enable pre-fetching accross epoch boundaries; 50% wall time
            # reduction if world_size ≈ len(self._paths)
            paths = itertools.cycle(paths)

        tokenize = MLMFileTokenizer(cfg, self._tokenizer)
        tokenize = functools.partial(
            mstar.utils.misc.wait_if_busy_fn,
            90, 2, 5, tokenize)
        # Limit #cpus used by hf/tokenizers CPUs for 20% wall time improvement
        tokenize = functools.partial(
            mstar.utils.misc.with_env_fn,
            {'RAYON_RS_NUM_CPUS': str(self._num_cpus)}, tokenize)

        ex = mstar.utils.executors.LazyProcessPoolExecutor(max_workers=1)
        processed = ex.map(tokenize, paths, prefetch=1)

        for shard in processed:
            if shard is None:
                continue
            yield from batch_iter_from_shard(
                shard, cfg.batch_size, self._shuffle, self._keep_last_batch,
                cfg.use_nsp
            )

    def __iter__(self):
        if self._eager:
            yield from self._eager_iter()
        else:
            yield from self._lazy_iter()
