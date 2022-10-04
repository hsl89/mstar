import os

from transformers import AlbertTokenizer
from .tokens import (
    MASK_TOKEN,
    SEP_TOKEN,
    CLS_TOKEN,
    SPM_UNK_TOKEN,
    SPM_PAD_TOKEN,
    SPIECE_UNDERLINE,
)

__all__ = ["M5SentencepieceTokenizer"]


VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}


class M5SentencepieceTokenizer(AlbertTokenizer):
    """
    Construct an M5 specific Sentencepiece tokenizer.
    Based on `SentencePiece <https://github.com/google/sentencepiece>`__.
    This tokenizer inherits from Huggingface transformers `AlbertTokenizer`
    which contains most of the main methods. Users should refer to this superclass
    for more information regarding those methods. The M5 Sentencepiece tokenizer
    differs from the sentencepiece tokenizer in this package only in the pre-processing.
    M5 Sentencepiece ignores Albert's pre-processing since it cannot be directly 
    serialized. This tokenizer assumes automatic lower casing, keeping white spaces and
    accents.

    Arguments:
        vocab_file (:obj:`str`):
            `SentencePiece <https://github.com/google/sentencepiece>`__ file
            (generally has a `.spm` extension) that contains the vocabulary
            necessary to instantiate a tokenizer.
        bos_token (:obj:`str`, `optional`, defaults to :obj:`"[CLS]"`):
            The beginning of sequence token that was used during pretraining.
            Can be used a sequence classifier token.
            .. note::
                When building a sequence using special tokens, this is not the token
                that is used for the beginning of sequence. The token used is the :obj:`cls_token`.
        eos_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
            The end of sequence token.
            .. note::
                When building a sequence using special tokens, this is not the token
                that is used for the end of sequence. The token used is the :obj:`sep_token`.
        unk_token (:obj:`str`, `optional`, defaults to :obj:`"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to
            an ID and is set to be this token instead.
        sep_token (:obj:`str`, `optional`, defaults to :obj:`"[SEP]"`):
            The separator token, which is used when building a sequence from multiple sequences,
            e.g. two sequences for sequence classification or for a text and a question for
            question answering. It is also used as the last token of a sequence
            built with special tokens.
        pad_token (:obj:`str`, `optional`, defaults to :obj:`"<pad>"`):
            The token used for padding, for example when batching sequences of different lengths.
        cls_token (:obj:`str`, `optional`, defaults to :obj:`"[CLS]"`):
            The classifier token which is used when doing sequence classification
            (classification of the whole sequence instead of per-token classification).
            It is the first token of the sequence when built with special tokens.
        mask_token (:obj:`str`, `optional`, defaults to :obj:`"[MASK]"`):
            The token used for masking values. This is the token used when training
            this model with masked language modeling.
            This is the token which the model will try to predict.
    Attributes:
        sp_model (:obj:`SentencePieceProcessor`):
         The `SentencePiece` processor that is used for every conversion (string, tokens and IDs).
    """

    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(
        self,
        vocab_file,
        bos_token=CLS_TOKEN,
        eos_token=SEP_TOKEN,
        unk_token=SPM_UNK_TOKEN,
        sep_token=SEP_TOKEN,
        pad_token=SPM_PAD_TOKEN,
        cls_token=CLS_TOKEN,
        mask_token=MASK_TOKEN,
        **kwargs
    ):
        if not os.path.isfile(vocab_file):
            raise ValueError("Can't find a vocabulary file at path '{}'.".format(vocab_file))

        super().__init__(
            vocab_file=vocab_file,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs
        )


    def _tokenize(self, text):
        """Tokenize a string."""
        text = self.preprocess_text(text)
        pieces = self.sp_model.encode(text, out_type=str)
        return pieces

    def preprocess_text(self, inputs):
        """
        Overriding the parent's class pre-processing to ensure consistency for all models.
        We are ignoring do_lower_case since it is automatic and redundant here.
        We are ignoring remove_space since it is not required.
        We are ignoring keep_accents since we always want to keep accents.
        """
        return inputs

    @staticmethod
    def is_first_subword(token):
        """Check if a string token is a subword following a previous subword,
        instead of the beginning of a word.
        """
        return token.startswith(SPIECE_UNDERLINE)

    @property
    def full_vocab_size(self):
        return len(self.get_vocab())
