"""
Collator for T5-style MLM. Modified from
https://github.com/huggingface/transformers/blob/c85547af2b69f9082bcd7bac97092b1d162f3fdc/examples/flax/language-modeling/run_t5_mlm_flax.py#L279
"""
import warnings
import torch
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
import numpy as np
import transformers
import random
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase
import logging

logger = logging.getLogger(__name__)

# CLM_TOKEN="<extra_id_1999>"
CLM_TOKEN = "<extra_id_0>"


def compute_input_and_target_lengths(
    inputs_length, noise_density, mean_noise_span_length
):
    """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2466>`__ .
    Training parameters to avoid padding with random_spans_noise_mask.
    When training a model with random_spans_noise_mask, we would like to set the other
    training hyperparmeters in a way that avoids padding.
    This function helps us compute these hyperparameters.
    We assume that each noise span in the input is replaced by extra_tokens_per_span_inputs sentinel tokens,
    and each non-noise span in the targets is replaced by extra_tokens_per_span_targets sentinel tokens.
    This function tells us the required number of tokens in the raw example (for split_tokens())
    as well as the length of the encoded targets. Note that this function assumes
    the inputs and targets will have EOS appended and includes that in the reported length.
    Args:
        inputs_length: an integer - desired length of the tokenized inputs sequence
        noise_density: a float
        mean_noise_span_length: a float
    Returns:
        tokens_length: length of original text in tokens
        targets_length: an integer - length in tokens of encoded targets sequence
    """

    def _tokens_length_to_inputs_length_targets_length(tokens_length):
        num_noise_tokens = int(round(tokens_length * noise_density))
        num_nonnoise_tokens = tokens_length - num_noise_tokens
        num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
        # inputs contain all nonnoise tokens, sentinels for all noise spans
        # and one EOS token.
        _input_length = num_nonnoise_tokens + num_noise_spans + 1
        _output_length = num_noise_tokens + num_noise_spans + 1
        return _input_length, _output_length

    tokens_length = inputs_length

    while (
        _tokens_length_to_inputs_length_targets_length(tokens_length + 1)[0]
        <= inputs_length
    ):
        tokens_length += 1

    inputs_length, targets_length = _tokens_length_to_inputs_length_targets_length(
        tokens_length
    )

    # minor hack to get the targets length to be equal to inputs length
    # which is more likely to have been set to a nice round number.
    if noise_density == 0.5 and targets_length > inputs_length:
        tokens_length -= 1
        targets_length -= 1
    return tokens_length, targets_length


# copied from transformers.models.bart.modeling_flax_bart.shift_tokens_right
def shift_tokens_right(
    input_ids: np.array, pad_token_id: int, decoder_start_token_id: int
) -> np.ndarray:
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = np.zeros_like(input_ids)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = decoder_start_token_id

    shifted_input_ids = np.where(
        shifted_input_ids == -100, pad_token_id, shifted_input_ids
    )
    return shifted_input_ids


class T5DataCollatorForSpanCorruption:

    """
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        expandend_inputs_length (:obj:`int`):
            The expected input length before masking.
            Should be greater than input_length
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        noise_density: float,
        mean_noise_span_length: float,
        input_length: int,
        expandend_inputs_length: int,
        target_length: int,
        pad_token_id: int,
        decoder_start_token_id: int,
    ):

        logger.warning(
            "Assumes that input data is packed via EOS token, don't use with unpacked data since this leads to padding token issues"
        )

        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length
        self.expandend_inputs_length = expandend_inputs_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(
        self, batch, return_type="pt", testing: bool = False
    ) -> Dict[str, np.ndarray]:
        # batch is a list of batch_size strings

        batch = self.tokenizer.batch_encode_plus(
            batch,
            return_token_type_ids=False,
            return_attention_mask=False,  # assumes examples are packed
            padding="max_length",
            truncation=True,
            max_length=self.expandend_inputs_length,
        )
        assert list(batch.keys()) == ["input_ids"], "Only want input ids"
        # convert list to dict and tensorize input
        # stack list of input ids to form [batch_size x max_length] matrix
        input_ids = np.stack(batch["input_ids"], axis=0)
        # original_input_ids = input_ids
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray(
            [
                self.random_spans_noise_mask(expandend_input_length)
                for i in range(batch_size)
            ]
        )

        # don't mask out T5 sentinel padding tokens, remove those from masking
        # mask_indices = np.where(input_ids==self.tokenizer.pad_token_id,False,mask_indices)
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)

        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be {self.target_length}."
            )

        # to check that tokens are correctly preprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.pad_token_id, self.decoder_start_token_id
        )

        # ignore all padding tokens in loss
        batch["labels"][batch["labels"] == self.pad_token_id] = -100

        if return_type == "pt":
            for key, val in batch.items():
                batch[key] = torch.tensor(val, dtype=torch.long)

        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )

        # skip first extra_id token which may be used for CLM with -1 below
        sentinel_ids = np.where(
            sentinel_ids != 0, (len(self.tokenizer) - 1 - sentinel_ids), 0
        )
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [
                input_ids,
                np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32),
            ],
            axis=-1,
        )
        return input_ids

    def random_spans_noise_mask(self, length):

        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .
        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.
        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number
        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))

        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(
            num_nonnoise_tokens, num_noise_spans
        )

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]


class BARTDataCollatorForSpanCorruption(T5DataCollatorForSpanCorruption):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        noise_density: float,
        mean_noise_span_length: float,
        input_length: int,
        expandend_inputs_length: int,
        target_length: int,
        pad_token_id: int,
        decoder_start_token_id: int,
        keep_sentinel_ids: bool = True,  # if true, keep sentinel id tokens in input, if false remove all sentinel id tokens
    ):

        super().__init__(
            tokenizer=tokenizer,
            noise_density=noise_density,
            mean_noise_span_length=mean_noise_span_length,
            input_length=input_length,
            expandend_inputs_length=expandend_inputs_length,
            target_length=target_length,
            pad_token_id=pad_token_id,
            decoder_start_token_id=decoder_start_token_id,
        )

        self.keep_sentinel_ids = keep_sentinel_ids

    def __call__(
        self, batch, return_type="pt", testing: bool = False
    ) -> Dict[str, np.ndarray]:
        # batch is a list of batch_size strings

        batch = self.tokenizer.batch_encode_plus(
            batch,
            return_token_type_ids=False,
            return_attention_mask=False,  # only works with packed examples
            padding="max_length",
            truncation=True,
            max_length=self.expandend_inputs_length,
        )
        assert list(batch.keys()) == ["input_ids"], "Only want input ids"
        # convert list to dict and tensorize input
        # stack list of input ids to form [batch_size x max_length] matrix
        input_ids = np.stack(batch["input_ids"], axis=0)

        # original_input_ids = input_ids
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray(
            [
                self.random_spans_noise_mask(expandend_input_length)
                for i in range(batch_size)
            ]
        )

        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))

        if self.keep_sentinel_ids:
            pass
        else:
            # remove all sentinel ids for more natural text generation
            input_ids_sentinel = np.where(input_ids_sentinel != 0, -1, 0)

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)

        # pad the labels (the un-altered input ids) to multiple of 128
        # right-pads last dimension with padding token value
        current_seq_length = input_ids.shape[-1]
        nearest_multiple = int(np.ceil(current_seq_length / 128) * 128)
        missing_length = nearest_multiple - current_seq_length
        padded_input_ids = np.pad(
            input_ids,
            (input_ids.ndim - 1) * [(0, 0)] + [(0, missing_length)],
            constant_values=self.tokenizer.pad_token_id,
        )
        # create decoder attention mask of same size
        batch["labels"] = padded_input_ids

        # batch["decoder_attention_mask"]=np.where(padded_input_ids==self.tokenizer.pad_token_id,0,1)

        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.pad_token_id, self.decoder_start_token_id
        )

        # ignore all padding tokens in loss
        batch["labels"][batch["labels"] == self.pad_token_id] = -100

        batch["attention_mask"] = np.where(
            batch["input_ids"] == self.tokenizer.pad_token_id, 0, 1
        )

        if return_type == "pt":
            for key, val in batch.items():
                batch[key] = torch.tensor(val, dtype=torch.long)
        elif return_type == "np":
            pass
        else:
            raise NotImplementedError()

        return batch


class Seq2SeqMixDenoisingCollator(transformers.DataCollatorForSeq2Seq):
    def __init__(
        self,
        tokenizer,
        max_length,
        pad_token_id,
        decoder_start_token_id,
        noise_density=0.15,
        mean_noise_span_length=3,
        keep_sentinel_ids=False,  # don't keep mask tokens in label/target
        mix_ratio=0.2,  # CLM ratio
        clm_max_doc=-1,  # cutoff for max docs per batch
        clm_min_length=128,
        clm_input_min_ratio=0.2,
        clm_input_max_ratio=0.8,
        clm_token=CLM_TOKEN,
        testing=False,  # checks that masking tokens are in the tokenizer special tokens
    ):
        super(Seq2SeqMixDenoisingCollator, self).__init__(tokenizer)
        # self.model = model
        self.tokenizer = tokenizer
        self.decoder_start_token_id = decoder_start_token_id
        self.max_length = max_length
        self.mix_ratio = mix_ratio
        self.clm_min_length = clm_min_length
        self.clm_max_doc = clm_max_doc
        self.clm_input_min_ratio = clm_input_min_ratio
        self.clm_input_max_ratio = clm_input_max_ratio
        self.testing = False
        self.clm_token = clm_token
        self.clm_max_doc = clm_max_doc

        self.bart_collator = BARTDataCollatorForSpanCorruption(
            tokenizer=tokenizer,
            noise_density=noise_density,
            mean_noise_span_length=mean_noise_span_length,
            input_length=max_length,
            expandend_inputs_length=max_length,
            target_length=0,
            pad_token_id=pad_token_id,
            decoder_start_token_id=decoder_start_token_id,
            keep_sentinel_ids=keep_sentinel_ids,
        )

    def __call__(self, examples, return_type="pt"):

        # first k are CLM, last n-k are BART
        k = int(self.mix_ratio * len(examples))

        batch = self.bart_collator(examples[k:], return_type="np")

        # clm batch
        if k > 0:
            # make sure that the tokenizer has the CLM token
            assert self.clm_token in self.tokenizer.vocab.keys()
            # only using
            clm_examples = examples[:k]
            # clm_examples = [self.tokenizer(x)['input_ids'] for x in clm_examples]
            # limit to first clm_max_doc documents atmost not to make batch large
            # assumes documents are packed using tokenizer.eos_token
            clm_examples = [
                self._half_seq(
                    t,
                    self.clm_token,
                    self.clm_input_min_ratio,
                    self.clm_input_max_ratio,
                )
                for e in examples[:k]
                for t in e.split(self.tokenizer.eos_token)
                if len(t.split()) >= self.clm_min_length
            ]

            clm_examples = clm_examples[: self.clm_max_doc]
            """
            for i,x in enumerate(clm_examples):
                print("Tuple ",type(x))
                for j,z in enumerate(x):
                    print("\tComponent",j)
                    print("\tstart",z[:50])
                    print("\tend",z[-50:])
            """

            if clm_examples:
                # avoid right-truncation for CLM continuation
                clm_batch = self.tokenizer(
                    [t[0] for t in clm_examples], verbose=False, truncation=False
                )
                clm_labels = self.tokenizer(
                    [t[1] for t in clm_examples],
                    truncation=True,
                    padding="max_length",
                    max_length=self.max_length,
                )
                """
                clm_batch = self.tokenizer.batch_encode_plus([t[0] for t in clm_examples],
                            padding='max_length',
                            truncation=True,
                            max_length=self.max_length)

                clm_labels = self.tokenizer.batch_encode_plus([t[1] for t in clm_examples],
                            padding='max_length',
                            truncation=True,
                            max_length=self.max_length)
                """
                """
                new_clm_labels = []
                for i,x in enumerate(clm_labels["input_ids"]):
                    missing_length = len(x)-self.max_length  
                    x = x+missing_length*[self.tokenizer.pad_token_id] 
                    new_clm_labels.append(x)
                """

                # clm_labels["input_ids"]=new_clm_labels

                # clm_batch["input_ids"] = new_clm_batch
                # cut off from left if too long and append CLM token
                self._truncate_from_left(clm_batch)
                new_clm_batch = []
                for i, x in enumerate(clm_batch["input_ids"]):
                    missing_length = self.max_length - len(x)
                    x = x + missing_length * [self.tokenizer.pad_token_id]
                    new_clm_batch.append(x)
                    # print(i,len(x))

                clm_batch["input_ids"] = new_clm_batch
                """
                for x in clm_batch["input_ids"]:
                    print(len(x),type(x))
                    print(x)
                """

                # stack clm input ids to form [batch_size x max_length] matrix
                clm_batch["input_ids"] = np.stack(
                    [np.array(x) for x in clm_batch["input_ids"]], axis=0
                )

                # pad input ids to max_length, since BART collator may not output the same max_length
                # padding is only on the RHS of the sequence dimension
                batch["input_ids"] = np.pad(
                    batch["input_ids"],
                    [(0, 0), (0, self.max_length - batch["input_ids"].shape[1])],
                    mode="constant",
                    constant_values=self.tokenizer.pad_token_id,
                )
                # stack CLM and span corruption along batch dim
                batch["input_ids"] = np.concatenate(
                    [batch["input_ids"], clm_batch["input_ids"]], axis=0
                )
                # mask encoder with 0's where there are pad tokens
                # skip decoder mask since decoder starts with pad token id by convention, and causal masking means we would only worry about intermediate padding tokens

                clm_labels["input_ids"] = np.stack(
                    [np.array(x) for x in clm_labels["input_ids"]], axis=0
                )
                batch["attention_mask"] = np.where(
                    batch["input_ids"] == self.tokenizer.pad_token_id, 0, 1
                )

                # print(batch["labels"].shape,clm_labels["input_ids"].shape)

                batch["labels"] = np.concatenate(
                    [batch["labels"], clm_labels["input_ids"]], axis=0
                )

                batch["labels"] = np.where(
                    batch["labels"] == self.tokenizer.pad_token_id,
                    -100,
                    batch["labels"],
                )

                # ok to just repeat the shift
                batch["decoder_input_ids"] = shift_tokens_right(
                    batch["labels"],
                    self.tokenizer.pad_token_id,
                    self.decoder_start_token_id,
                )

        if return_type == "pt":
            for key, val in batch.items():
                batch[key] = torch.tensor(val, dtype=torch.long)

        return batch

    @staticmethod
    def _sample_length(prob_dist="poisson"):
        if prob_dist == "poisson":
            return np.random.poisson(3)  # bart paper
        if prob_dist == "geometric":
            return np.random.geometric(0.2)  # spanBERT paper

    @staticmethod
    def _half_seq(example, clm_token, input_min_ratio=0.2, input_max_ratio=0.8):
        # print(type(example))
        # print(example)
        # raise ValueError
        tokens = example.split()
        # alexaTM comments below
        # originally 0.2 to 0.8 for 20B
        # make it 0 to 0.9 for 100B
        random_split = np.random.uniform(input_min_ratio, input_max_ratio)
        l = int(len(tokens) * random_split)
        # add clm token to signal to the model to do CLM
        if l == 0:
            return clm_token, " ".join(tokens[l:])
        else:
            return clm_token + " " + " ".join(tokens[:l]), " ".join(tokens[l:])

    def _truncate_from_left(self, batch):
        for i in range(len(batch["input_ids"])):
            if len(batch["input_ids"][i]) > self.max_length:
                extra = len(batch["input_ids"][i]) - self.max_length
                for key in batch.keys():
                    # make sure don't cut <s>, _, and clm_token
                    batch[key][i] = batch[key][i][0:3] + batch[key][i][extra + 3 :]


class MixedT5DataCollatorForSpanCorruption(transformers.DataCollatorForSeq2Seq):

    """
    Mixes CLM objective with Data collator used for T5 span-masked language modeling.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        expandend_inputs_length (:obj:`int`):
            The expected input length before masking.
            Should be greater than input_length
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        noise_density: float,
        mean_noise_span_length: float,
        input_length: int,
        expandend_inputs_length: int,
        target_length: int,
        pad_token_id: int,
        decoder_start_token_id: int,
        clm_ratio: float,
        clm_max_doc: int,
        clm_max_output_length: int,
        clm_token: str = CLM_TOKEN,
    ):
        super(MixedT5DataCollatorForSpanCorruption, self).__init__(tokenizer)

        assert (
            target_length < clm_max_output_length
        ), "CLM output length should be longer than T5 output length for concat later"
        self.tokenizer = tokenizer
        self.clm_ratio = clm_ratio
        self.clm_max_output_length = clm_max_output_length
        self.decoder_start_token_id = decoder_start_token_id

        self.t5_data_collator = T5DataCollatorForSpanCorruption(
            tokenizer=tokenizer,
            noise_density=noise_density,
            mean_noise_span_length=mean_noise_span_length,
            input_length=input_length,
            expandend_inputs_length=expandend_inputs_length,
            target_length=target_length,
            pad_token_id=pad_token_id,
            decoder_start_token_id=decoder_start_token_id,
        )

        self.clm_data_collator = CLMCollator(
            tokenizer=tokenizer,
            max_input_length=input_length,
            max_output_length=clm_max_output_length,
            decoder_start_token_id=decoder_start_token_id,
            clm_token=clm_token,
            clm_max_doc=clm_max_doc,
        )

    def __call__(
        self, batch, return_type="pt", testing: bool = False
    ) -> Dict[str, np.ndarray]:
        # batch is a list with length batch_size and entries are strings

        # return each collator values in numpy, then possibly convert to 'pt' at end
        INTERMEDIATE_RETURN_TYPE = "np"

        assert (
            self.clm_ratio == 0.5
        ), "Easy to extend, but this was faster since we can split the batch size"
        # where to split batch
        split_position = len(batch) // 2

        # in case examples are not shuffled by dataloader
        random.shuffle(batch)

        clm_examples = batch[:split_position]
        t5_examples = batch[split_position:]

        # split batch

        t5_batch = self.t5_data_collator(
            batch=t5_examples, return_type=INTERMEDIATE_RETURN_TYPE
        )  # INTERMEDIATE_RETURN_TYPE)

        # don't truncate t5 targets, assumes T5 output length is shorter and will be extended to match CLM output length
        assert (
            t5_batch["labels"].shape[1] < self.clm_data_collator.max_output_length
        ), "Don't truncate T5 loss labels, increase CLM output length"

        clm_batch = self.clm_data_collator(
            examples=clm_examples, return_type=INTERMEDIATE_RETURN_TYPE
        )

        if clm_batch == {}:
            # no examples meet length cutoff, empty dict
            batch = {}

            batch["input_ids"] = t5_batch["input_ids"]
            # re-does clm attention mask, but not a major timing concern
            batch["attention_mask"] = np.where(
                batch["input_ids"] == self.tokenizer.pad_token_id, 0, 1
            )

            padded_t5_labels = np.pad(
                t5_batch["labels"],
                [(0, 0), (0, self.clm_max_output_length - t5_batch["labels"].shape[1])],
                mode="constant",
                constant_values=self.tokenizer.pad_token_id,
            )

            batch["labels"] = padded_t5_labels
            batch["labels"] = np.where(
                batch["labels"] == self.tokenizer.pad_token_id, -100, batch["labels"]
            )
            # ok to just repeat the shift, though this effort could be saved
            batch["decoder_input_ids"] = shift_tokens_right(
                batch["labels"],
                self.tokenizer.pad_token_id,
                self.decoder_start_token_id,
            )

            if return_type == "pt":
                for key, val in batch.items():
                    batch[key] = torch.tensor(val, dtype=torch.long)

            return batch

        else:
            batch = {}

            batch["input_ids"] = np.concatenate(
                [t5_batch["input_ids"], clm_batch["input_ids"]], axis=0
            )
            # re-does clm attention mask, but not a major timing concern
            batch["attention_mask"] = np.where(
                batch["input_ids"] == self.tokenizer.pad_token_id, 0, 1
            )

            padded_t5_labels = np.pad(
                t5_batch["labels"],
                [(0, 0), (0, self.clm_max_output_length - t5_batch["labels"].shape[1])],
                mode="constant",
                constant_values=self.tokenizer.pad_token_id,
            )

            # print(t5_batch["labels"].shape)
            # print(padded_t5_labels.shape)
            # print(clm_batch["labels"].shape)

            batch["labels"] = np.concatenate(
                [padded_t5_labels, clm_batch["labels"]], axis=0
            )
            batch["labels"] = np.where(
                batch["labels"] == self.tokenizer.pad_token_id, -100, batch["labels"]
            )
            # ok to just repeat the shift, though this effort could be saved
            batch["decoder_input_ids"] = shift_tokens_right(
                batch["labels"],
                self.tokenizer.pad_token_id,
                self.decoder_start_token_id,
            )

            if return_type == "pt":
                for key, val in batch.items():
                    batch[key] = torch.tensor(val, dtype=torch.long)

            return batch


class CLMCollator(transformers.DataCollatorForSeq2Seq):
    def __init__(
        self,
        tokenizer,
        max_input_length: int,
        max_output_length: int,
        decoder_start_token_id: int,
        clm_token: str,
        clm_min_split_ratio: float = 0.2,  # maximum leftmost split cutoff from alexatm
        clm_max_split_ratio: float = 0.8,  # maximum rightmost split cutoff from alextm
        clm_min_length: int = 128,  # discard examples that are shorter than this, cutoff taken from alexatm
        clm_max_doc: int = -1,  # max number to return
        greedy_pack: bool = False,
    ):
        super().__init__(tokenizer)
        # self.model = model
        self.tokenizer = tokenizer
        self.decoder_start_token_id = decoder_start_token_id
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.clm_token = clm_token
        self.clm_min_split_ratio = clm_min_split_ratio
        self.clm_max_split_ratio = clm_max_split_ratio
        self.clm_min_length = clm_min_length
        self.greedy_pack = greedy_pack
        self.clm_max_doc = clm_max_doc

    def __call__(self, examples: list, return_type="pt"):
        # examples is a list of strings of text that are pre-packed using tokenizer eos token

        # make sure that the tokenizer has the CLM token
        assert self.clm_token in self.tokenizer.vocab.keys()

        # uses whitespace to split, which avoids mid-token splits for text generation
        # assumes documents are packed using tokenizer.eos_token
        split_clm_examples = [
            self._split_seq(
                t, self.clm_token, self.clm_min_split_ratio, self.clm_max_split_ratio
            )
            for e in examples
            for t in e.split(self.tokenizer.eos_token)
            if len(t.split()) >= self.clm_min_length
        ]
        if self.clm_max_doc != -1:
            split_clm_examples = split_clm_examples[: self.clm_max_doc]

        # no examples pass, return empty dict
        if len(split_clm_examples) == 0:
            return {}

        # left truncation for max input length done later
        # left truncation because this is a CLM input

        clm_inputs = self.tokenizer(
            [t[0] for t in split_clm_examples], verbose=False, truncation=False
        )

        # right truncation for max output length
        clm_labels = self.tokenizer(
            [t[1] for t in split_clm_examples],
            truncation=True,
            padding="max_length",
            max_length=self.max_output_length,
        )

        if self.greedy_pack:
            # pack examples in greedy manner
            # inputs are self.clm_token ....... self.tokenizer.eos_token self.clm_token ....... self.tokenizer.eos_token
            # outputs are self.clm_token ....... self.tokenizer.eos_token self.clm_token ....... self.tokenizer.eos_token
            raise NotImplementedError
        else:
            # assumes passed batch size is target batch size
            # print(type(clm_inputs['input_ids']))
            # print(clm_inputs['input_ids'])
            # for x in clm_inputs['input_ids']:
            #    print(len(x))
            # print(clm_inputs)
            # raise ValueError(type(clm_inputs))
            # print(clm_inputs.keys())
            for key in clm_inputs.keys():
                clm_inputs[key] = clm_inputs[key][: len(examples)]
            for key in clm_labels.keys():
                clm_labels[key] = clm_labels[key][: len(examples)]
                # print(key,type(clm_inputs[key]))
                # print(key,clm_inputs[key][0])

        # print(type(clm_inputs))
        # raise ValueError
        # need to do this no matter what, truncate left to best length
        # for i,x in enumerate(clm_inputs['input_ids']):
        #    print(i,len(x))

        self._truncate_from_left(clm_inputs)

        new_clm_batch = []
        for i, x in enumerate(clm_inputs["input_ids"]):
            # print(x,type(x),len(x),self.max_input_length)
            missing_length = self.max_input_length - len(x)
            x = x + missing_length * [self.tokenizer.pad_token_id]
            new_clm_batch.append(x)

        clm_inputs["input_ids"] = new_clm_batch

        # stack clm input ids to form [batch_size x max_length] matrix
        clm_inputs["input_ids"] = np.stack(
            [np.array(x) for x in clm_inputs["input_ids"]], axis=0
        )

        # construct batch from dict
        batch = {}
        # pad input ids to max_length, since BART collator may not output the same max_length
        # padding is only on the RHS of the sequence dimension
        batch["input_ids"] = np.pad(
            clm_inputs["input_ids"],
            [(0, 0), (0, self.max_input_length - clm_inputs["input_ids"].shape[1])],
            mode="constant",
            constant_values=self.tokenizer.pad_token_id,
        )

        # mask encoder with 0's where there are pad tokens
        # skip decoder mask since decoder starts with pad token id by convention, and causal masking means we would only worry about intermediate padding tokens

        clm_labels["input_ids"] = np.stack(
            [np.array(x) for x in clm_labels["input_ids"]], axis=0
        )
        batch["attention_mask"] = np.where(
            batch["input_ids"] == self.tokenizer.pad_token_id, 0, 1
        )

        batch["labels"] = clm_labels["input_ids"]

        # mask the padding tokens
        batch["labels"] = np.where(
            batch["labels"] == self.tokenizer.pad_token_id, -100, batch["labels"]
        )

        # ok to just repeat the shift
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.tokenizer.pad_token_id, self.decoder_start_token_id
        )

        if return_type == "pt":
            for key, val in batch.items():
                batch[key] = torch.tensor(val, dtype=torch.long)
        return batch

    @staticmethod
    def _sample_length(prob_dist="poisson"):
        if prob_dist == "poisson":
            return np.random.poisson(3)  # bart paper
        if prob_dist == "geometric":
            return np.random.geometric(0.2)  # spanBERT paper

    @staticmethod
    def _split_seq(
        example: str, clm_token: str, input_min_ratio: float, input_max_ratio: float
    ) -> tuple:
        """
        Split input example into a tuple (input, output)
        """
        # print(type(example))
        # print(example)
        # raise ValueError
        tokens = example.split()
        # alexaTM comments below
        # originally 0.2 to 0.8 for 20B
        random_split = np.random.uniform(input_min_ratio, input_max_ratio)
        l = int(len(tokens) * random_split)
        # add clm token to signal to the model to do CLM
        if l == 0:
            return clm_token, " ".join(tokens[l:])
        else:
            return clm_token + " " + " ".join(tokens[:l]), " ".join(tokens[l:])

    def _truncate_from_left(self, batch):
        # print(batch)#["input_ids"])
        # raise ValueError
        for i in range(len(batch["input_ids"])):
            if len(batch["input_ids"][i]) > self.max_input_length:
                extra = len(batch["input_ids"][i]) - self.max_input_length
                for key in batch.keys():
                    # make sure don't cut <s>, _, and clm_token
                    # print(batch[key][i][extra + 3:])
                    batch[key][i] = batch[key][i][0:3] + batch[key][i][extra + 3 :]
