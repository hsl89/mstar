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
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
)


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


class UnpackedT5DataCollatorForSpanCorruption:

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
        decoder_start_token_id: int,
    ):

        self.tokenizer = tokenizer
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.input_length = input_length
        self.expandend_inputs_length = expandend_inputs_length
        self.target_length = target_length
        self.decoder_start_token_id = decoder_start_token_id
        assert self.tokenizer.pad_token_id is not None

    # def __call__(self, examples: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    def __call__(
        self, batch, return_type="pt", testing: bool = False
    ) -> Dict[str, np.ndarray]:
        # batch is a list of batch_size strings

        batch = self.tokenizer.batch_encode_plus(
            batch,
            return_token_type_ids=False,
            return_attention_mask=False,
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

        assert mask_indices.shape == input_ids.shape
        # print("Mask indices last 10 cols")
        # print(mask_indices.sum()/np.prod(mask_indices.shape))
        # print(mask_indices[:,-10:])
        # don't want to apply MLM objective to padding tokens
        # print("Input ids last 10 cols")
        # print(input_ids[:,-10:])
        # mask_indices = np.where(input_ids!=self.tokenizer.pad_token_id,mask_indices,False)
        # print("Expect this to decrease since we no longer mask padding tokens")
        # print(mask_indices.sum()/np.prod(mask_indices.shape))

        # no sentinel masking of padding tokens
        mask_indices = np.where(
            input_ids != self.tokenizer.pad_token_id, mask_indices, False
        )
        labels_mask = ~mask_indices
        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        # mask_indices = np.where(input_ids==self.tokenizer.pad_token_id,False,mask_indices)
        batch["input_ids"] = []
        batch["labels"] = []
        for idx in range(batch_size):

            # print(input_ids_sentinel[idx:idx+1].shape)
            # print("Sentinel ids First 10 cols")
            # print(input_ids_sentinel[idx:idx+1,:10])
            # print("Sentinel ids last 10 cols")
            # print(input_ids_sentinel[idx:idx+1,-10:])

            # account for padding tokens we didn't mask
            batch["input_ids"].append(
                self.filter_input_ids(
                    input_ids[idx : idx + 1], input_ids_sentinel[idx : idx + 1]
                )
            )
            # print(batch["input_ids"][idx].shape)
            # print("Should be ",self.input_length, "is longer, see above, full input see below")
            # print(input_ids.shape)
            # raise ValueError

            batch["labels"].append(
                self.filter_input_ids(
                    input_ids[idx : idx + 1], labels_sentinel[idx : idx + 1]
                )
            )

            # print(batch["labels"][idx].shape)
            # print("Should be ",self.target_length, "is shorter, see above")#, full input see below")
            # print(input_ids.shape)

        # all input_ids  should be <= target length since we don't span-subtract padding tokens
        # print([x.shape for x in batch["input_ids"]])
        assert all([x.shape[-1] >= self.input_length for x in batch["input_ids"]])
        # print([x.shape for x in batch["labels"]])
        # all labels should be <= target length since we don't span-subtract padding tokens
        assert all([x.shape[-1] <= self.target_length for x in batch["labels"]])

        def pad_up_to_length(array_list, length):
            pass

        # pad labels up to full expected length
        padding_blocks = [
            self.tokenizer.pad_token_id * np.ones([1, self.target_length - x.shape[-1]])
            for x in batch["labels"]
        ]
        # print("percent label padding",sum([x.shape[-1] for x in padding_blocks])/(len(batch["labels"])*self.target_length))

        batch["labels"] = np.concatenate(
            [
                np.concatenate([x, y], axis=-1)
                for x, y in zip(batch["labels"], padding_blocks)
            ],
            axis=0,
        )
        batch["labels"] = np.where(
            batch["labels"] == self.tokenizer.pad_token_id, -100, batch["labels"]
        )
        ##ignore all padding tokens in loss
        # batch['labels'][batch['labels']==self.tokenizer.pad_token_id]=-100

        # -100 padding tokens from labels are converted back to pad_token_id, so decoder_attn_mask must use pad_token_id
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.tokenizer.pad_token_id, self.decoder_start_token_id
        )
        # -100 padding tokens are pad_token_id again due to shift_tokens_right
        batch["decoder_attention_mask"] = np.where(
            batch["decoder_input_ids"] == self.tokenizer.pad_token_id, False, True
        )

        # need to avoid masking the decoder start token
        batch["decoder_attention_mask"][:, 0] = True

        # print("Percent decoder input padding",1-batch["decoder_attention_mask"].sum()/np.prod(batch["decoder_attention_mask"].shape))

        # truncate input_ids down to full length
        batch["input_ids"] = np.concatenate(
            [x[:, : self.input_length] for x in batch["input_ids"]], axis=0
        )
        # print(batch["input_ids"].shape)
        # for x in batch['input_ids']:
        #    print(x.shape)
        batch["attention_mask"] = np.where(
            batch["input_ids"] == self.tokenizer.pad_token_id, 0.0, 1.0
        )
        # print(batch["attention_mask"].sum()/(512*6))
        # raise ValueError

        # to check that tokens are correctly preprocessed, one can run `self.tokenizer.batch_decode(input_ids)` and `self.tokenizer.batch_decode(labels)` here...

        # assuems that labels are already padded and concatenated

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be {self.target_length}."
            )

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
