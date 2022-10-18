from typing import Optional, Union

import copy
import json
import random
import re
from dataclasses import dataclass

import torch
import numpy as np

from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    PreTrainedTokenizerBase,
)

try:
    from PIL import Image
except:
    pass

"""
batch_size = 1000 samples
bs = 10 -> 100 -> 500 -> 1000
batch_size = 2
2 x 1024
4 x 512
8 x 256
16 x 128
--------
8 x 128

100 x 256
50 x 512
25 x 1024

1 x 16
1 x 32 ( 2 x 16)
1 x 64
2 x 64 ( 4 x 16)


"""

"""
2 x 1024
---
16 x 128
"""

# class WaelDataloader:
#
#     def __iter__(self):
#         for batch in self.wrapped_data_loader:
#             # split batch in batches[4] -> 4 x 128
#             for micro_batch in batches:
#                 q.push(micro_batch)
#                 yield q.pop()
#             # yield batch
#             # batch 16 x 128

# causal language masking task token
CLM_TOKEN = "<extra_id_1999>"
# extra id token string to be formatted
FORMAT_STR = "<extra_id_{}>"


class CollatorForWholeWorkMasking(DataCollatorForWholeWordMask):
    def __init__(self, device, max_length, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.max_length = max_length

    def set_max_length(self, l):
        pass

    def __call__(self, examples):
        batch = self.tokenizer(
            [e["text"] for e in examples],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
        )
        batch = list(
            map(dict, zip(*[[(k, v) for v in batch[k]] for k in batch.keys()]))
        )

        batch = super().__call__(batch)
        for k, v in batch.items():
            batch[k] = v.to(self.device)
        return batch


class CollatorForLMWrapper(DataCollatorForLanguageModeling):
    def __init__(self, device, max_length, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.max_length = max_length

    def set_max_length(self, l):
        pass

    def __call__(self, examples):
        batch = self.tokenizer(
            [e["text"] for e in examples],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
        )
        batch = list(
            map(dict, zip(*[[(k, v) for v in batch[k]] for k in batch.keys()]))
        )

        batch = super().__call__(batch)
        for k, v in batch.items():
            batch[k] = v.to(self.device)
        return batch


# copied partly from huggingface
@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`,
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        # max_label_length = max(len(l) for l in features['labels'])

        max_label_length = self.max_length
        padding_side = self.tokenizer.padding_side
        labels = []
        for feature in features["labels"]:
            remainder = [self.label_pad_token_id] * (max_label_length - len(feature))
            labels.append(
                feature + remainder if padding_side == "right" else remainder + feature
            )

        features["labels"] = labels

        if "decoder_input_ids" in features.keys():
            dii = []
            for feature in features["decoder_input_ids"]:
                remainder = [self.tokenizer.pad_token_id] * (
                    max_label_length - len(feature)
                )
                dii.append(
                    feature + remainder
                    if padding_side == "right"
                    else remainder + feature
                )

            features["decoder_input_ids"] = dii
        if "input_ids" in features.keys():
            features = self.tokenizer.pad(
                features,
                padding="max_length",
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
        ## This is to support the case the encoder is ViT
        else:
            for k, v in features.items():
                if not torch.is_tensor(v):
                    features[k] = torch.tensor(v)

        # if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
        #     decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
        #     features["decoder_input_ids"] = decoder_input_ids

        return features


class Seq2SeqDenoisingCollator(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, max_length, device, mask_fraction=0.15, no_mask=True):
        super(Seq2SeqDenoisingCollator, self).__init__(tokenizer)
        # self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_fraction = mask_fraction
        self.mask_id = tokenizer.mask_token_id
        self.device = device
        self.no_mask = no_mask

    def __call__(self, examples):

        # padding will be done
        batch = self.tokenizer(
            [e["text"] for e in examples], truncation=True, max_length=self.max_length
        )
        batch["labels"] = copy.deepcopy(batch["input_ids"])
        # TODO: support token_type_ids if needed

        def valid(start, length, l):
            for i in range(length):
                if start + i in l:
                    continue
                else:
                    return False
            return True

        if "token_type_ids" in batch.keys():
            masked_inputs = []
            masked_inputs_attention_mask = []
            masked_token_type_ids = []
            for example, attention_mask, token_type_id in zip(
                batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]
            ):
                length = len(example)

                # decide how many tokens to mask
                num_masked = max(
                    int(float(length) * self.mask_fraction + 0.5), 1
                )  # masking at least 1

                # sample the length of span need to be masked, 3 is the number in the BART paper
                span_lengths = []
                while num_masked > 0:
                    sample_length = self._sample_length()
                    span_length = (
                        sample_length if sample_length <= num_masked else num_masked
                    )
                    span_lengths.append(span_length)
                    num_masked -= span_length

                # sample the start index of span for each span length sampled from the above
                # don't mask <bos>
                masked_idx = list(range(1, length))
                zero_nums = 0
                for l in span_lengths:
                    # get the valid start indexes
                    candidates = [i for i in masked_idx if valid(i, l, masked_idx)]
                    start = random.sample(candidates, 1)[0]
                    if l != 0:
                        example[start] = self.mask_id
                        for i in range(l):
                            masked_idx.remove(start + i)
                            if i != 0 or self.no_mask:
                                example[start + i] = -1
                                token_type_id[start + i] = -1
                    else:
                        zero_nums += 1
                masked_input = list(filter(lambda x: x != -1, example))
                masked_token_type_id = list(filter(lambda x: x != -1, token_type_id))

                masked_inputs.append(masked_input)
                masked_inputs_attention_mask.append([1] * len(masked_input))
                masked_token_type_ids.append(masked_token_type_id)

            batch["input_ids"] = masked_inputs
            batch["attention_mask"] = masked_inputs_attention_mask
            batch["token_type_ids"] = masked_token_type_ids
        else:
            masked_inputs = []
            masked_inputs_attention_mask = []
            for example, attention_mask in zip(
                batch["input_ids"], batch["attention_mask"]
            ):
                length = len(example)

                # decide how many tokens to mask
                num_masked = max(
                    int(float(length) * self.mask_fraction + 0.5), 1
                )  # masking at least 1

                # sample the length of span need to be masked, 3 is the number in the BART paper
                span_lengths = []
                while num_masked > 0:
                    sample_length = self._sample_length()
                    span_length = (
                        sample_length if sample_length <= num_masked else num_masked
                    )
                    span_lengths.append(span_length)
                    num_masked -= span_length

                # sample the start index of span for each span length sampled from the above
                # don't mask <bos>
                masked_idx = list(range(1, length))
                zero_nums = 0
                for l in span_lengths:
                    # get the valid start indexes
                    candidates = [i for i in masked_idx if valid(i, l, masked_idx)]
                    start = random.sample(candidates, 1)[0]
                    if l != 0:
                        example[start] = self.mask_id
                        for i in range(l):
                            masked_idx.remove(start + i)
                            if i != 0 or self.no_mask:
                                example[start + i] = -1
                    else:
                        zero_nums += 1
                masked_input = list(filter(lambda x: x != -1, example))
                masked_inputs.append(masked_input)
                masked_inputs_attention_mask.append([1] * len(masked_input))

            batch["input_ids"] = masked_inputs
            batch["attention_mask"] = masked_inputs_attention_mask
        batch = super().__call__(batch)
        for k, v in batch.items():
            batch[k] = v.to(self.device)

        return batch

    @staticmethod
    def _sample_length(prob_dist="poisson"):
        if prob_dist == "poisson":
            return np.random.poisson(3)  # bart paper
        if prob_dist == "geometric":
            return np.random.geometric(0.2)  # spanBERT paper


class Seq2SeqMixDenoisingCollator(DataCollatorForSeq2Seq):
    def __init__(
        self,
        tokenizer,
        max_length,
        device,
        mask_fraction=0.15,
        no_mask=True,  # don't keep mask tokens in label/target
        mix_ratio=0.2,
        clm_min_length=128,
        clm_max_doc=1,
        clm_input_min_ratio=0.2,
        clm_input_max_ratio=0.8,
        testing=False,  # checks that masking tokens are in the tokenizer special tokens
    ):
        super(Seq2SeqMixDenoisingCollator, self).__init__(tokenizer)
        # self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_fraction = mask_fraction
        self.mask_id = tokenizer.mask_token_id
        self.device = device
        self.no_mask = no_mask
        if not self.no_mask:
            assert (
                self.mask_id is not None
            ), "Need a mask token instantiated for BART masking"
        self.mix_ratio = mix_ratio
        self.clm_min_length = clm_min_length
        self.clm_max_doc = clm_max_doc
        self.clm_input_min_ratio = clm_input_min_ratio
        self.clm_input_max_ratio = clm_input_max_ratio
        self.testing = False

    def __call__(self, examples):

        # padding will be done
        k = int(self.mix_ratio * len(examples))

        # denoising batch
        batch = self.tokenizer(
            [e["text"] for e in examples[k:]],
            truncation=True,
            max_length=self.max_length,
        )
        batch["labels"] = copy.deepcopy(batch["input_ids"])
        # TODO: support token_type_ids if needed

        def valid(start, length, l):
            for i in range(length):
                if start + i in l:
                    continue
                else:
                    return False
            return True

        print("Starting masking")
        if "token_type_ids" in batch.keys():
            raise NotImplementedError("This has not been rigorously inspected")
            masked_inputs = []
            masked_inputs_attention_mask = []
            masked_token_type_ids = []
            for example, attention_mask, token_type_id in zip(
                batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]
            ):
                length = len(example)

                # decide how many tokens to mask
                num_masked = max(
                    int(float(length) * self.mask_fraction + 0.5), 1
                )  # masking at least 1

                # sample the length of span need to be masked, 3 is the number in the BART paper
                span_lengths = []
                while num_masked > 0:
                    sample_length = self._sample_length()
                    span_length = (
                        sample_length if sample_length <= num_masked else num_masked
                    )
                    span_lengths.append(span_length)
                    num_masked -= span_length

                # sample the start index of span for each span length sampled from the above
                # don't mask <bos>
                masked_idx = list(range(1, length))
                zero_nums = 0
                for l in span_lengths:
                    # get the valid start indexes
                    candidates = [i for i in masked_idx if valid(i, l, masked_idx)]
                    start = random.sample(candidates, 1)[0]
                    if l != 0:
                        example[start] = self.mask_id
                        for i in range(l):
                            masked_idx.remove(start + i)
                            if i != 0 or self.no_mask:
                                example[start + i] = -1
                                token_type_id[start + i] = -1
                    else:
                        zero_nums += 1
                masked_input = list(filter(lambda x: x != -1, example))
                masked_token_type_id = list(filter(lambda x: x != -1, token_type_id))

                masked_inputs.append(masked_input)
                masked_inputs_attention_mask.append([1] * len(masked_input))
                masked_token_type_ids.append(masked_token_type_id)

            batch["input_ids"] = masked_inputs
            batch["attention_mask"] = masked_inputs_attention_mask
            batch["token_type_ids"] = masked_token_type_ids
        else:
            masked_inputs = []
            masked_inputs_attention_mask = []
            for example, attention_mask in zip(
                batch["input_ids"], batch["attention_mask"]
            ):
                length = len(example)
                # decide how many tokens to mask
                num_masked = max(
                    int(float(length) * self.mask_fraction + 0.5), 1
                )  # masking at least 1

                # sample the length of span need to be masked, 3 is the number in the BART paper
                span_lengths = []
                while num_masked > 0:
                    sample_length = self._sample_length()
                    span_length = (
                        sample_length if sample_length <= num_masked else num_masked
                    )
                    span_lengths.append(span_length)
                    num_masked -= span_length

                # sample the start index of span for each span length sampled from the above
                # don't mask <bos>
                masked_idx = list(range(1, length))
                zero_nums = 0
                for l in span_lengths:
                    # get the valid start indexes
                    candidates = [i for i in masked_idx if valid(i, l, masked_idx)]
                    start = random.sample(candidates, 1)[0]
                    if l != 0:
                        example[start] = self.mask_id
                        for i in range(l):
                            masked_idx.remove(start + i)
                            if i != 0 or self.no_mask:
                                example[start + i] = -1
                    else:
                        zero_nums += 1
                masked_input = list(filter(lambda x: x != -1, example))
                masked_inputs.append(masked_input)
                masked_inputs_attention_mask.append([1] * len(masked_input))

            batch["input_ids"] = masked_inputs
            batch["attention_mask"] = masked_inputs_attention_mask
        print("Starting clm")
        # clm batch
        if k > 0:
            # limit to first clm_max_doc documents atmost not to make batch large
            # assumes documents are packed using tokenizer.eos_token
            clm_examples = [
                self._half_seq(t, self.clm_input_min_ratio, self.clm_input_max_ratio)
                for e in examples[:k]
                for t in e["text"].split(self.tokenizer.eos_token)[: self.clm_max_doc]
                if len(t.split()) >= self.clm_min_length
            ]
            if clm_examples:
                clm_batch = self.tokenizer([t[0] for t in clm_examples], verbose=False)
                clm_labels = self.tokenizer(
                    [t[1] for t in clm_examples],
                    truncation=True,
                    max_length=self.max_length,
                )

                self._truncate_from_left(clm_batch)
                batch["input_ids"].extend(clm_batch["input_ids"])
                batch["attention_mask"].extend(clm_batch["attention_mask"])
                if "token_type_ids" in batch.keys():
                    batch["token_type_ids"].extend(clm_batch["token_type_ids"])
                batch["labels"].extend(clm_labels["input_ids"])

        batch = super().__call__(batch)
        for k, v in batch.items():
            batch[k] = v.to(self.device)

        return batch

    @staticmethod
    def _sample_length(prob_dist="poisson"):
        if prob_dist == "poisson":
            return np.random.poisson(3)  # bart paper
        if prob_dist == "geometric":
            return np.random.geometric(0.2)  # spanBERT paper

    @staticmethod
    def _half_seq(example, input_min_ratio=0.2, input_max_ratio=0.8):
        tokens = example.split()
        # originally 0.2 to 0.8 for 20B
        # make it 0 to 0.9 for 100B
        random_split = np.random.uniform(input_min_ratio, input_max_ratio)
        l = int(len(tokens) * random_split)
        # add CLM_TOKEN to signal to the model to do CLM
        if l == 0:
            return CLM_TOKEN, " ".join(tokens[l:])
        else:
            return CLM_TOKEN + " " + " ".join(tokens[:l]), " ".join(tokens[l:])

    def _truncate_from_left(self, batch):
        for i in range(len(batch["input_ids"])):
            if len(batch["input_ids"][i]) > self.max_length:
                extra = len(batch["input_ids"][i]) - self.max_length
                for key in batch.keys():
                    # make sure don't cut <s>, _, and RESERVED_1999
                    batch[key][i] = batch[key][i][0:3] + batch[key][i][extra + 3 :]


class Seq2SeqDenoisingT5Collator(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, max_length, device, mask_fraction=0.15):
        super(Seq2SeqDenoisingT5Collator, self).__init__(tokenizer)
        # self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_fraction = mask_fraction
        self.mask_id = tokenizer.mask_token_id
        self.device = device
        self.tokenizer = tokenizer

    def __call__(self, examples):

        # padding will be done
        batch = self.tokenizer(
            [e["text"] for e in examples], truncation=True, max_length=self.max_length
        )

        # TODO: support token_type_ids if needed

        def valid(start, length, l):
            for i in range(length):
                if start + i in l:
                    continue
                else:
                    return False
            return True

        if "token_type_ids" in batch.keys():
            masked_inputs = []
            masked_inputs_attention_mask = []
            masked_token_type_ids = []
            labels = []
            for example, attention_mask, token_type_id in zip(
                batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]
            ):
                length = len(example)

                # decide how many tokens to mask
                num_masked = max(
                    int(float(length) * self.mask_fraction + 0.5), 1
                )  # masking at least 1

                # sample the length of span need to be masked, 3 is the number in the BART paper
                span_lengths = []
                while num_masked > 0:
                    sample_length = self._sample_length()
                    span_length = (
                        sample_length if sample_length <= num_masked else num_masked
                    )
                    span_lengths.append(span_length)
                    num_masked -= span_length

                # sample the start index of span for each span length sampled from the above
                # don't mask <bos>
                masked_idx = list(range(1, length))
                zero_nums = 0
                # counter = -1
                # label_counter = -1
                label = list()
                # label.append(self.tokenizer.convert_tokens_to_ids('RESERVED_{}'.format(-label_counter)))
                label.append(-2000)
                for l in span_lengths:
                    # get the valid start indexes
                    candidates = [i for i in masked_idx if valid(i, l, masked_idx)]
                    start = random.sample(candidates, 1)[0]
                    if l != 0:
                        label.append(example[start])
                        # example[start] = self.tokenizer.convert_tokens_to_ids('RESERVED_{}'.format(-counter))
                        example[start] = -2000
                        for i in range(l):
                            masked_idx.remove(start + i)
                            if i != 0:
                                label.append(example[start + i])
                                example[start + i] = -1000
                                token_type_id[start + i] = -1000
                        # label_counter -= 1
                        # counter -= 1
                        # label.append(self.tokenizer.convert_tokens_to_ids('RESERVED_{}'.format(-label_counter)))
                        label.append(-2000)
                    else:
                        zero_nums += 1
                masked_input = list(filter(lambda x: x != -1000, example))
                masked_token_type_id = list(filter(lambda x: x != -1000, token_type_id))

                counter = -1
                for i in range(len(masked_input)):
                    if masked_input[i] == -2000:
                        token = FORMAT_STR.format(-counter)
                        if self.testing:
                            assert (
                                token
                                in self.tokenizer.special_tokens_map[
                                    "additional_special_tokens"
                                ]
                            )
                        masked_input[i] = self.tokenizer.convert_tokens_to_ids(token)
                        counter -= 1

                counter = -1
                for i in range(len(label)):
                    if label[i] == -2000:
                        token = FORMAT_STR.format(-counter)
                        if self.testing:
                            assert (
                                token
                                in self.tokenizer.special_tokens_map[
                                    "additional_special_tokens"
                                ]
                            )
                        label[i] = self.tokenizer.convert_tokens_to_ids(token)
                        counter -= 1

                masked_inputs.append(masked_input)
                masked_inputs_attention_mask.append([1] * len(masked_input))
                masked_token_type_ids.append(masked_token_type_id)
                labels.append(label)

            batch["input_ids"] = masked_inputs
            batch["attention_mask"] = masked_inputs_attention_mask
            batch["token_type_ids"] = masked_token_type_ids
            batch["labels"] = labels
        else:
            masked_inputs = []
            masked_inputs_attention_mask = []
            labels = []
            for example, attention_mask in zip(
                batch["input_ids"], batch["attention_mask"]
            ):
                length = len(example)

                # decide how many tokens to mask
                num_masked = max(
                    int(float(length) * self.mask_fraction + 0.5), 1
                )  # masking at least 1

                # sample the length of span need to be masked, 3 is the number in the BART paper
                span_lengths = []
                while num_masked > 0:
                    sample_length = self._sample_length()
                    span_length = (
                        sample_length if sample_length <= num_masked else num_masked
                    )
                    span_lengths.append(span_length)
                    num_masked -= span_length

                # sample the start index of span for each span length sampled from the above
                # don't mask <bos>
                masked_idx = list(range(1, length))
                zero_nums = 0
                # counter = -1
                # label_counter = -1
                label = list()
                # label.append(self.tokenizer.convert_tokens_to_ids('RESERVED_{}'.format(-label_counter)))
                label.append(-2000)
                for l in span_lengths:
                    # get the valid start indexes
                    candidates = [i for i in masked_idx if valid(i, l, masked_idx)]
                    start = random.sample(candidates, 1)[0]
                    if l != 0:
                        label.append(example[start])
                        # example[start] = self.tokenizer.convert_tokens_to_ids('RESERVED_{}'.format(-counter))
                        example[start] = -2000
                        for i in range(l):
                            masked_idx.remove(start + i)
                            if i != 0:
                                label.append(example[start + i])
                                example[start + i] = -1000
                        # label_counter -= 1
                        # counter -= 1
                        # label.append(self.tokenizer.convert_tokens_to_ids('RESERVED_{}'.format(-label_counter)))
                        label.append(-2000)
                    else:
                        zero_nums += 1
                masked_input = list(filter(lambda x: x != -1000, example))

                counter = -1
                for i in range(len(masked_input)):
                    if masked_input[i] == -2000:
                        token = FORMAT_STR.format(-counter)
                        if self.testing:
                            assert (
                                token
                                in self.tokenizer.special_tokens_map[
                                    "additional_special_tokens"
                                ]
                            )
                        masked_input[i] = self.tokenizer.convert_tokens_to_ids(token)
                        counter -= 1

                counter = -1
                for i in range(len(label)):
                    if label[i] == -2000:
                        token = FORMAT_STR.format(-counter)
                        if self.testing:
                            assert (
                                token
                                in self.tokenizer.special_tokens_map[
                                    "additional_special_tokens"
                                ]
                            )
                        label[i] = self.tokenizer.convert_tokens_to_ids(token)
                        counter -= 1

                masked_inputs.append(masked_input)
                masked_inputs_attention_mask.append([1] * len(masked_input))
                labels.append(label)

            batch["input_ids"] = masked_inputs
            batch["attention_mask"] = masked_inputs_attention_mask
            # batch['token_type_ids'] = masked_token_type_ids
            batch["labels"] = labels

        batch = super().__call__(batch)
        for k, v in batch.items():
            batch[k] = v.to(self.device)

        return batch

    @staticmethod
    def _sample_length(prob_dist="poisson"):
        if prob_dist == "poisson":
            return np.random.poisson(3)  # bart paper
        if prob_dist == "geometric":
            return np.random.geometric(0.2)  # spanBERT paper


class Seq2SeqFinetuningCollator(DataCollatorForSeq2Seq):
    def __init__(
        self,
        tokenizer,
        max_length,
        device,
        add_clm,
        truncate_from_left,
        clm_token=CLM_TOKEN,
    ):
        super(Seq2SeqFinetuningCollator, self).__init__(tokenizer)
        # self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.add_clm = add_clm
        self.truncate_from_left = truncate_from_left
        # make sure clm token is already added
        assert clm_token in tokenizer.special_tokens_map["additional_special_tokens"]
        assert (
            clm_token == CLM_TOKEN
        ), f"Not using predefined CLM token {CLM_TOKEN} may be very hazardous"
        self.clm_token = clm_token

    def __call__(self, examples):
        if self.add_clm:
            src = [self.clm_token + " " + e["src"] for e in examples]
        else:
            src = [e["src"] for e in examples]
        tgt = [e["tgt"] for e in examples]
        # padding will be done
        if self.truncate_from_left:
            batch = self.tokenizer(src, verbose=False)
            self._truncate_from_left(batch)
        else:
            batch = self.tokenizer(src, truncation=True, max_length=self.max_length)
        labels = self.tokenizer(tgt, truncation=True, max_length=self.max_length)

        batch["labels"] = labels["input_ids"]

        batch = super().__call__(batch)
        for k, v in batch.items():
            batch[k] = v.to(self.device)

        return batch

    def _truncate_from_left(self, batch):
        if self.add_clm:
            keep_from_left = 3
        else:
            keep_from_left = 1
        for i in range(len(batch["input_ids"])):
            if len(batch["input_ids"][i]) > self.max_length:
                extra = len(batch["input_ids"][i]) - self.max_length + keep_from_left
                for key in batch.keys():
                    # make sure don't cut <s>, _, and RESERVED_1999
                    batch[key][i] = (
                        batch[key][i][0:keep_from_left]
                        + batch[key][i][extra + keep_from_left :]
                    )
