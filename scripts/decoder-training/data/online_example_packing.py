import os
import torch
import numpy as np
from torch.utils.data import IterableDataset

class ExamplePackDataset(IterableDataset):
    def __init__(self, dataset, dataset_indices, tokenizer, batch_size, max_seq_length, seed, partition, max_batch):
        self.dataset = dataset
        self.dataset_indices = dataset_indices
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        # batch size in terms of number of tokens
        self.batch_size_token = batch_size* max_seq_length
        # total number of samples to be processed by 1 gpu
        self.length = len(dataset_indices)
        # the index of the sample to read from the dataset
        self.current_index = 0
        # buffer of one or more samples before tokenization, used after reading long sequences
        self.buffer = []
        self.buffer_length = 0
        # buffer of extra tokens generated after tokenization, compared to batch_size_token
        self.buffer_token = []
        self.buffer_token_length = 0
        self.partition = partition
        self.seed = seed
        self.current_epoch = 0
        self.process_global_rank = torch.distributed.get_rank()
        self.batch_index = 0
        self.max_batch = max_batch
        # max number of characters in a sequence that can be processed by tokenzier without any issues
        self.max_char_length = 500000

    def _randomize_index(self):
        seed = self.seed + self.current_epoch
        rng = np.random.default_rng(seed)
        rng.shuffle(self.dataset_indices)
        print('\ngpu: {} - finished {}-th epoch of data.\n'.format(self.process_global_rank, self.current_epoch))
        return

    def _split_sample(self, sample):
        # split the input sample into multiple samples of length self.max_char_length
        # or less (in case of last sample)
        n_chars = len(sample)
        n_sample = int(n_chars/self.max_char_length)
        sample_list = []
        for i in range(n_sample):
            sample_list.append(sample[i* self.max_char_length : (i+1)* self.max_char_length])
        # if more words left, append as last sample
        if n_sample* self.max_char_length < n_chars:
            sample_list.append(sample[n_sample* self.max_char_length :])
        return sample_list

    def _read_samples(self):
        # read sample with at least self.batch_size_token number of words,
        # in excess of (self.buffer_length + self.buffer_token_length)
        n_word = self.buffer_length + self.buffer_token_length
        samples = []
        while n_word < self.batch_size_token:
            if self.partition == 'train':
                # when we reach the end of the dataset, update self.current_epoch and randomize
                if self.current_index >= self.length:
                    self.current_epoch += 1
                    self.current_index = 0
                    self._randomize_index()
            else:
                # when we reach the end of the dataset, stop iterating
                if self.current_index >= self.length:
                    self.current_index = 0
                # changing the condition to avoid having NCCL timeout error. It happens because each gpu gets different
                # number of batches due to sample based data split (equal no of samples does not imply equal number of
                # batches). This problem is severe with more splits of dev set, because of the smaller size in each set.
                if self.batch_index >= self.max_batch:
                    self.batch_index = 0
                    raise StopIteration
            current_sample = self.dataset.take([self.dataset_indices[self.current_index]])["text"].to_pylist()[0]
            self.current_index += 1

            current_len = len(current_sample)
            if current_len > self.max_char_length:
                # split current_sample into multiple samples before appending
                sample_list = self._split_sample(current_sample)
                for seq in sample_list:
                    # add the splits to samples until self.batch_size_token words are in
                    # samples, then add rest of them to the self.buffer for use in the
                    # following steps (in case of long documents)
                    if n_word < self.batch_size_token:
                        samples.append(seq)
                        n_word += len(seq.split())
                    else:
                        self.buffer.append(seq)
                        self.buffer_length += len(seq.split())
            else:
                samples.append(current_sample)
                # count the number of words in the current sample
                n_word += len(current_sample.split())
        #print('current number of words: {}'.format(n_word))

        return samples

    def _tokenize_samples(self, samples):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # tokenize the input text samples
        batch = self.tokenizer(samples,
                        return_attention_mask=False,
                        return_token_type_ids=False,
                        truncation=False)

        # if self.buffer_token is not empty, start from there
        if self.buffer_token_length > 0:
            input_ids = self.buffer_token
        else:
            input_ids = []

        # add bos_token and eos_token around each sample, and concatenate
        for ids in batch["input_ids"]:
            # Gopher style
            #ids = [self.tokenizer.bos_token_id] + ids + [self.tokenizer.eos_token_id]
            # GPT3 style
            ids = ids + [self.tokenizer.eos_token_id]
            input_ids += ids

        #print('current number of tokens: {}'.format(len(input_ids)))
        # check if we have enough tokens to construct a batch
        assert len(input_ids) >= self.batch_size_token

        # keep self.batch_size_token tokens in input_ids and keep the
        # extra tokens in self.buffer_token to use in next step
        self.buffer_token = input_ids[self.batch_size_token: ]
        self.buffer_token_length = len(self.buffer_token)
        input_ids = input_ids[0: self.batch_size_token]
        return input_ids

    def _create_batch(self, input_ids):
        input_ids = torch.LongTensor(input_ids)
        input_ids = input_ids.reshape(self.batch_size, self.max_seq_length)
        batch = {}
        batch["input_ids"] = input_ids
        batch["attention_mask"] = torch.ones(input_ids.size(), dtype=torch.int64)
        batch["labels"] = batch["input_ids"].clone()
        if self.partition == 'train':
            batch["current_index"] = self.current_index
            batch["current_epoch"] = self.current_epoch

        return batch

    def __next__(self):

        # read samples according to batch size, (check length of each sample, if larger than threshold
        # split into multiple samples) tokenize, concatenate, check if equal or more than desired batch
        # size (in number of tokens). use batch_size number of tokens to create a batch (make tensor, add
        # an extra dimension as done by tokenizer) and keep the rest in buffer. optionally use the buffer
        # in the following batch or discard it if less that number of tokens_per_sample

        if self.partition!='train' and self.batch_index >= self.max_batch:
            self.batch_index = 0
            raise StopIteration

        # if the current buffer doesn't have enough tokens to form a batch, read new samples
        if (self.buffer_length + self.buffer_token_length) < self.batch_size_token:
            samples = self._read_samples()
            input_ids = self._tokenize_samples(self.buffer + samples)
            batch = self._create_batch(input_ids)
            self.buffer = []
            self.buffer_length = 0

        # if self.buffer_token has enough tokens to create one or few batches, use
        # them here to create batches one at a time. this happens because we don't
        # have good estimate on how many samples to read for a batch.
        elif self.buffer_token_length >= self.batch_size_token:
            batch = self._create_batch(self.buffer_token[:self.batch_size_token])
            self.buffer_token = self.buffer_token[self.batch_size_token:]
            self.buffer_token_length -= self.batch_size_token

        # (rare case: when we have encountered a very large sequence in the last step)
        else:
            if len(self.buffer) > self.batch_size:
                batch = self._tokenize_samples(self.buffer[0:self.batch_size])
                self.buffer = self.buffer[self.batch_size:]
                self.buffer_length -= self.batch_size_token
            else:
                batch = self._tokenize_samples(self.buffer)
                self.buffer = []
                self.buffer_length = 0

        self.batch_index += 1
        return batch

    def __iter__(self):
        return self


