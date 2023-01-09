# coding=utf-8
# Copyright 2021 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# FIXME need to update this file to make encoding work. Currently we are only using the Fast tokenizer which already works.
import os
import sys
import itertools
from contextlib import contextmanager
from functools import lru_cache
from shutil import copyfile
from typing import Any, Dict, List, Optional, Tuple

import regex as re
import sentencepiece as spm
from transformers.tokenization_utils import (
    AddedToken,
    BatchEncoding,
    PreTrainedTokenizer,
    TextInput,
    _is_end_of_word,
    _is_start_of_word,
)
from transformers.utils import logging

from consts import FAIRSEQ_LANGUAGE_CODES, USER_DEFINED_SYMBOLS
from pretokenize import gpt2_pretokenize_pattern

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.
    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    greek = ['Γ', 'Δ', 'Θ', 'Λ', 'Ξ', 'Π', 'Σ', 'Φ', 'Ψ', 'Ω'] \
            + [chr(c) for c in range(ord('α'), ord('λ'))] \
            + [chr(c) for c in range(ord('π'), ord('ω'))]
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1

    cs = [chr(n) for n in cs]
    d = dict(zip(bs, cs))
    exceptions = [144, 145, 157, 158, 168, 170, 175, 178, 179, 180, 181, 184, 185, 186, 188, 189, 190]
    for i in range(len(exceptions)):
        d[exceptions[i]] = greek[i]
    return d


logger = logging.get_logger(__name__)

SPIECE_UNDERLINE = "▁"

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "vectorbert-large": "./sentencepiece.bpe.model",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "vectorbart-large": 1024,
}


class VectorBartTokenizer(PreTrainedTokenizer):
    """
    Construct a MBart50 tokenizer. Based on `SentencePiece <https://github.com/google/sentencepiece>`__.
    This tokenizer inherits from :class:`~transformers.PreTrainedTokenizer` which contains most of the main methods.
    Users should refer to this superclass for more information regarding those methods.
    Args:
        vocab_file (:obj:`str`):
            Path to the vocabulary file.
        src_lang (:obj:`str`, `optional`):
            A string representing the source language.
        tgt_lang (:obj:`str`, `optional`):
            A string representing the target language.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The end of sequence token.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"</s>"`):
            The separator token, which is used when building a sequence from multiple sequences, e.g. two sequences for
            sequence classification or for a text and a question for question answering. It is also used as the last
            token of a sequence built with special tokens.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"<s>"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special tokens.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"<mask>"`):
            The token used for masking values. This is the token used when training this model with masked language
            modeling. This is the token which the model will try to predict.
        sp_model_kwargs (:obj:`dict`, `optional`):
            Will be passed to the ``SentencePieceProcessor.__init__()`` method. The `Python wrapper for SentencePiece
            <https://github.com/google/sentencepiece/tree/master/python>`__ can be used, among other things, to set:
            - ``enable_sampling``: Enable subword regularization.
            - ``nbest_size``: Sampling parameters for unigram. Invalid for BPE-Dropout.
              - ``nbest_size = {0,1}``: No sampling is performed.
              - ``nbest_size > 1``: samples from the nbest_size results.
              - ``nbest_size < 0``: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.
            - ``alpha``: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.
        fix_trimmed_whitespace (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Set to `True` to ensure that newlines, tabs, and whitespace surrounding special
            tokens are not removed.
    Examples::
        >>> from transformers import MBart50Tokenizer
        >>> tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50", src_lang="en_XX", tgt_lang="ro_RO")
        >>> src_text = " UN Chief Says There Is No Military Solution in Syria"
        >>> tgt_text =  "Şeful ONU declară că nu există o soluţie militară în Siria"
        >>> model_inputs = tokenizer(src_text, return_tensors="pt")
        >>> with tokenizer.as_target_tokenizer():
        ...    labels = tokenizer(tgt_text, return_tensors="pt").input_ids
        >>> # model(**model_inputs, labels=labels) should work
    """

    vocab_files_names = VOCAB_FILES_NAMES
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]

    prefix_tokens: List[int] = []
    suffix_tokens: List[int] = []

    def __init__(
        self,
        vocab_file,
        src_lang=None,
        tgt_lang=None,
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        fix_trimmed_whitespace=False,
        use_phrases=True,
        **kwargs,
    ) -> None:
        # Mask token behave like a normal word, i.e. include the space before it
        mask_token = (
            AddedToken(mask_token, lstrip=True, rstrip=False)
            if isinstance(mask_token, str)
            else mask_token
        )

        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        super().__init__(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            sp_model_kwargs=self.sp_model_kwargs,
            **kwargs,
        )

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(str(vocab_file))
        self.vocab_file = vocab_file

        self.phrase_dict = None
        self.phrase_ids_to_tokens = {}
        self.phrase_tokens_to_ids = {}

        self.fix_trimmed_whitespace = fix_trimmed_whitespace
        self._new_tokens = None
        # how big was the original tokenizer? to do: find a clever way to compute this.
        self.original_sp_model_size = len(self.sp_model) # robkwiat@ :  Why can we not just call len?

        # Original fairseq vocab and spm vocab must be "aligned":
        # Vocab    |    0    |    1    |   2    |    3    |  4  |  5  |  6  |   7   |   8   |  9
        # -------- | ------- | ------- | ------ | ------- | --- | --- | --- | ----- | ----- | ----
        # fairseq  | '<s>'   | '<pad>' | '</s>' | '<unk>' | ',' | '.' | '▁' | 's'   | '▁de' | '-'
        # spm      | '<unk>' | '<s>'   | '</s>' | ','     | '.' | '▁' | 's' | '▁de' | '-'   | '▁a'

        # The first "real" token "," has position 4 in the original fairseq vocab and position 3 in the spm vocab
        self.fairseq_offset = 1
        # compute offset for first position of other groups of tokens
        self.sp_model_size = len(self.sp_model) - len(self.new_tokens)
        base_offset = self.sp_model_size + self.fairseq_offset
        mask_offset = base_offset + len(FAIRSEQ_LANGUAGE_CODES)
        new_token_offset = mask_offset + 1

        # Mimic fairseq token-to-id alignment for the first 4 token
        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
        self.fairseq_tokens_to_ids["<mask>"] = mask_offset

        self.lang_code_to_id = {
            code: i + base_offset
            for i, code in enumerate(FAIRSEQ_LANGUAGE_CODES)
        }
        for i, code in enumerate(self.new_tokens):
            self.lang_code_to_id[code] = i + new_token_offset
        new_token_offset = new_token_offset + len(self.new_tokens)

        if use_phrases:
            import json
            vocab_dir = vocab_file[:-len('sentencepiece.bpe.model')]
            try:
                merged_phrases = json.load(open(vocab_dir+'merged_phrases.json', 'r'))
                i = 0
                self.phrase_dict = {}
                for phrase in merged_phrases:
                    pretoks = phrase.split('▁▁▁')
                    phrase_joined = ''.join(pretoks)
                    self.phrase_tokens_to_ids[phrase_joined] = i + new_token_offset
                    self.phrase_ids_to_tokens[i + new_token_offset] = phrase_joined
                    sub_d = self.phrase_dict
                    for pretok in pretoks:
                        if pretok not in sub_d:
                            sub_d[pretok] = {}
                        sub_d = sub_d[pretok]
                    # sub_d[None] = len(self.sp_model) + i
                    sub_d[None] = phrase_joined
                    i += 1
                print('Loaded Merged Phrases')
            except:
                self.phrase_dict = None
                self.phrase_ids_to_tokens = {}
                self.phrase_tokens_to_ids = {}
                print('Failed to Load Merged Phrases')


        self.id_to_lang_code = {v: k for k, v in self.lang_code_to_id.items()}

        self.fairseq_tokens_to_ids.update(self.lang_code_to_id)
        self.fairseq_ids_to_tokens = {
            v: k for k, v in self.fairseq_tokens_to_ids.items()
        }
        self._additional_special_tokens = (
                list(self.lang_code_to_id.keys()) + USER_DEFINED_SYMBOLS
        )

        self._src_lang = src_lang if src_lang is not None else "<en_XX>"
        self.cur_lang_code_id = self.lang_code_to_id[self._src_lang]
        self.tgt_lang = tgt_lang
        self.set_src_lang_special_tokens(self._src_lang)

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.pat = gpt2_pretokenize_pattern
        # self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\n+|\t+|\s+(?!\S)| +|\s+""")

        # both variables below are used for constrained generation only
        self.vocab_trie = None
        self.single_space_mask = None

    @property
    def new_tokens(self) -> List[str]:
        if self._new_tokens is None:
            # Find all new tokens at the end of the vocabulary:
            existing_specials = FAIRSEQ_LANGUAGE_CODES + USER_DEFINED_SYMBOLS
            self._new_tokens = [
                self.sp_model.id_to_piece(token_id)
                for token_id in range(self.sp_model.get_piece_size())
                if token_id >= self.original_sp_model_size
                   and self.sp_model.id_to_piece(token_id) not in existing_specials
            ]

        return self._new_tokens

    @property
    def vocab_size(self) -> int:
        return (
            len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset + 1 - len(self.new_tokens) + len(self.phrase_ids_to_tokens)
        )  # Plus 1 for the mask token, new tokens are already counted in sp_model

    @property
    def src_lang(self) -> str:
        return self._src_lang

    @src_lang.setter
    def src_lang(self, new_src_lang: str) -> None:
        self._src_lang = new_src_lang
        self.set_src_lang_special_tokens(self._src_lang)

    def __getstate__(self) -> Dict:
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d: Dict) -> None:
        self.__dict__ = d

        # for backward compatibility
        if not hasattr(self, "sp_model_kwargs"):
            self.sp_model_kwargs = {}

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    def get_vocab(self) -> Dict:
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def tokenize(self, text: TextInput, **kwargs) -> List[str]:
        if self.fix_trimmed_whitespace:
            return self._safe_tokenize(text, **kwargs)
        else:
            return super().tokenize(text, **kwargs)

    def _safe_tokenize(self, text: TextInput, **kwargs) -> List[str]:
        """
        Override default huggingface: tokenize() method
        Result: Preserve newlines, tabs, and whitespace next to special tokens
                during huggingface's pre-tokenization.

        Converts a string in a sequence of tokens, using the tokenizer.

        Split in words for word-based vocabulary or sub-words for sub-word-based vocabularies
        (BPE/SentencePieces/WordPieces). Takes care of added tokens.

        Args:
            text (:obj:`str`):
                The sequence to be encoded.
            **kwargs (additional keyword arguments):
                Passed along to the model-specific ``prepare_for_tokenization`` preprocessing method.

        Returns:
            :obj:`List[str]`: The list of tokens.
        """
        # Simple mapping string => AddedToken for special tokens with specific tokenization behaviors
        all_special_tokens_extended = dict(
            (str(t), t)
            for t in self.all_special_tokens_extended
            if isinstance(t, AddedToken)
        )

        text, kwargs = self.prepare_for_tokenization(text, **kwargs)

        if kwargs:
            logger.warning(f"Keyword arguments {kwargs} not recognized.")

        if hasattr(self, "do_lower_case") and self.do_lower_case:
            # convert non-special tokens to lowercase
            escaped_special_toks = [
                re.escape(s_tok) for s_tok in self.all_special_tokens
            ]
            pattern = r"(" + r"|".join(escaped_special_toks) + r")|" + r"(.+?)"
            text = re.sub(
                pattern, lambda m: m.groups()[0] or m.groups()[1].lower(), text
            )

        def split_on_token(tok, text):
            result = []
            tok_extended = all_special_tokens_extended.get(tok, None)
            split_text = text.split(tok)
            full_word = ""
            for i, sub_text in enumerate(split_text):
                # AddedToken can control whitespace stripping around them.
                # We use them for GPT2 and Roberta to have different behavior depending on the special token
                # Cf. https://github.com/huggingface/transformers/pull/2778
                # and https://github.com/huggingface/transformers/issues/3788
                if isinstance(tok_extended, AddedToken):
                    if tok_extended.single_word:
                        # Try to avoid splitting on token
                        if (
                                i < len(split_text) - 1
                                and not _is_end_of_word(sub_text)
                                and not _is_start_of_word(split_text[i + 1])
                        ):
                            # Don't extract the special token
                            full_word += sub_text + tok
                        elif full_word:
                            full_word += sub_text
                            result.append(full_word)
                            full_word = ""
                            continue
                    # Strip white spaces on the right
                    if tok_extended.rstrip and i > 0:
                        # A bit counter-intuitive but we strip the left of the string
                        # since tok_extended.rstrip means the special token is eating all white spaces on its right
                        sub_text = sub_text.lstrip(" ")
                    # Strip white spaces on the left
                    if tok_extended.lstrip and i < len(split_text) - 1:
                        sub_text = sub_text.rstrip(" ")  # Opposite here

                # MODIFICATION: REMOVE ELSE STATEMENT
                # Result: for all standard tok_extended *do not* strip any \n, \t or whitespace
                #         instead, let the tokenizer preserve these characters

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self._tokenize(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def _tokenize(self, text: str) -> List[str]:
        tokenized = []
        pre_tokens = []
        for token in re.findall(self.pat, text):
            preencoded_token = "".join(
                [self.byte_encoder[c] for c in token.encode("utf-8", errors="replace")]
            )
            pre_tokens.append(preencoded_token)

        i = 0
        while i < len(pre_tokens):
            # Phrase Logic
            if self.phrase_dict is not None and pre_tokens[i] in self.phrase_dict:
                sub_d, j = self.phrase_dict, 0
                while i+j < len(pre_tokens) and pre_tokens[i+j] in sub_d:
                    sub_d = sub_d[pre_tokens[i+j]]
                    j += 1
                if None in sub_d:
                    tokenized.append(sub_d[None])
                    i = i+j
                    continue
            # End Phrase Logic

            preencoded_token = pre_tokens[i]
            encoded_token = self.sp_model.encode(preencoded_token, out_type=str)
            # skip a single `_` token before the special token to save spaces.
            # This only applies to the case when tokenizing a single special token where SPM will add a SPIECE_UNDERLINE subtoken before.
            if encoded_token[0] == SPIECE_UNDERLINE:
                encoded_token = encoded_token[1:]
            tokenized.extend(encoded_token)
            i += 1
        return tokenized

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a token (str) in an id using the vocab."""
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        elif token in self.phrase_tokens_to_ids:
            return self.phrase_tokens_to_ids[token]
        spm_id = self.sp_model.PieceToId(token)

        # Need to return unknown token if the SP model returned 0
        return spm_id + self.fairseq_offset if spm_id else self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        elif index in self.phrase_ids_to_tokens:
            return self.phrase_ids_to_tokens[index]
        return self.sp_model.IdToPiece(index - self.fairseq_offset)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        text = "".join(tokens).replace(" ", "")
        # FIXME the current tokenizer may not work for non-EN chars. Proceed for now as we focus on EN only.
        text = bytearray(
            [
                self.byte_decoder[c]
                for c in text.replace(SPIECE_UNDERLINE, "")
                if c in self.byte_decoder
            ]
        ).decode("utf-8", errors="replace")
        return text

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "")
            + VOCAB_FILES_NAMES["vocab_file"],
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` method.
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not the token list is already formatted with special tokens for the model.
        Returns:
            :obj:`List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """

        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        prefix_ones = [1] * len(self.prefix_tokens)
        suffix_ones = [1] * len(self.suffix_tokens)
        if token_ids_1 is None:
            return prefix_ones + ([0] * len(token_ids_0)) + suffix_ones
        return (
            prefix_ones
            + ([0] * len(token_ids_0))
            + ([0] * len(token_ids_1))
            + suffix_ones
        )

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An MBART-50 sequence has the following format, where ``X`` represents the sequence:
        - ``input_ids`` (for encoder) ``[src_lang_code] X [eos]``
        - ``labels``: (for decoder) ``[tgt_lang_code] X [eos]``
        BOS is never used. Pairs of sequences are not the expected use case, but they will be handled without a
        separator.
        Args:
            token_ids_0 (:obj:`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (:obj:`List[int]`, `optional`):
                Optional second list of IDs for sequence pairs.
        Returns:
            :obj:`List[int]`: List of `input IDs <../glossary.html#input-ids>`__ with the appropriate special tokens.
        """
        if token_ids_1 is None:
            return self.prefix_tokens + token_ids_0 + self.suffix_tokens
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return self.prefix_tokens + token_ids_0 + token_ids_1 + self.suffix_tokens

    def _build_translation_inputs(
        self,
        raw_inputs,
        src_lang: Optional[str],
        tgt_lang: Optional[str],
        **extra_kwargs,
    ):
        """Used by translation pipeline, to prepare inputs for the generate function"""
        if src_lang is None or tgt_lang is None:
            raise ValueError(
                "Translation requires a `src_lang` and a `tgt_lang` for this model"
            )
        self.src_lang = src_lang
        inputs = self(
            raw_inputs, add_special_tokens=True, return_tensors="pt", **extra_kwargs
        )
        tgt_lang_id = self.convert_tokens_to_ids(tgt_lang)
        inputs["forced_bos_token_id"] = tgt_lang_id
        return inputs

    def prepare_seq2seq_batch(
        self,
        src_texts: List[str],
        src_lang: str = "en_XX",
        tgt_texts: Optional[List[str]] = None,
        tgt_lang: str = "ro_RO",
        **kwargs,
    ) -> BatchEncoding:
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        return super().prepare_seq2seq_batch(src_texts, tgt_texts, **kwargs)

    @contextmanager
    def as_target_tokenizer(self):
        """
        Temporarily sets the tokenizer for encoding the targets. Useful for tokenizer associated to
        sequence-to-sequence models that need a slightly different processing for the labels.
        """
        self.set_tgt_lang_special_tokens(self.tgt_lang)
        yield
        self.set_src_lang_special_tokens(self.src_lang)

    def set_src_lang_special_tokens(self, src_lang: str) -> None:
        """Reset the special tokens to the source lang setting. prefix=[src_lang_code] and suffix=[eos]."""
        self.cur_lang_code_id = self.lang_code_to_id[src_lang]
        self.prefix_tokens = [self.cur_lang_code_id]
        self.suffix_tokens = []  # [self.eos_token_id]

    def set_tgt_lang_special_tokens(self, tgt_lang: str) -> None:
        """Reset the special tokens to the target language setting. prefix=[tgt_lang_code] and suffix=[eos]."""
        self.cur_lang_code_id = self.lang_code_to_id[tgt_lang]
        self.prefix_tokens = []  # [self.cur_lang_code_id]
        self.suffix_tokens = []  # [self.eos_token_id]
