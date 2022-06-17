import os
from shutil import copyfile
from transformers import MT5TokenizerFast
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast, PreTrainedTokenizerBase
from transformers.convert_slow_tokenizer import SpmConverter
from tokenizers import processors
import sentencepiece as spm

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

# Not using tokenizer

class ATMConverter(SpmConverter):
    def __init__(self, *args, create_type_ids=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.create_type_ids = create_type_ids

    def post_processor(self):
        if self.create_type_ids:
            return processors.TemplateProcessing(
                single=["<s>:0", "$A:0", "</s>:0"],
                pair=["<s>:0", "$A:0", "</s>:0", "$B:1", "</s>:1"],
                special_tokens=[
                    ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),
                    ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
                ],
            )
        else:
            return processors.TemplateProcessing(
                single=["<s>", "$A", "</s>"],
                pair=["<s>", "$A", "</s>", "$B", "</s>"],
                special_tokens=[
                    ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),
                    ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
                ],
            )


class ATMTokenizerFast(PreTrainedTokenizerFast):

    vocab_files_names = VOCAB_FILES_NAMES

    class ATMTokenizer(PreTrainedTokenizer):
        def __init__(
                self,
                vocab_file,
                bos_token="<s>",
                eos_token="</s>",
                unk_token="<unk>",
                sep_token="</s>",
                pad_token="[PAD]",
                cls_token="<s>",
                mask_token="[MASK]",
                **kwargs
        ):
            super().__init__(
                bos_token=bos_token,
                eos_token=eos_token,
                unk_token=unk_token,
                sep_token=sep_token,
                pad_token=pad_token,
                cls_token=cls_token,
                mask_token=mask_token,
                **kwargs
            )
            self.vocab_file = vocab_file
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.Load(self.vocab_file)

        def convert_tokens_to_ids(self, token):
            return self.sp_model.piece_to_id(token)

    # pylint: disable=non-parent-init-called
    def __init__(self,
                 vocab_file,
                 bos_token="<s>",
                 eos_token="</s>",
                 unk_token="<unk>",
                 sep_token="</s>",
                 pad_token="[PAD]",
                 cls_token="<s>",
                 mask_token="[MASK]",
                 create_segment_ids=False,
                 **kwargs):
        self.vocab_file = vocab_file
        self.create_segment_ids = create_segment_ids
        slow_tokenizer = ATMTokenizerFast.ATMTokenizer(vocab_file, **kwargs)
        self._tokenizer = self.create_fast_tokenizer(slow_tokenizer, create_segment_ids)
        PreTrainedTokenizerBase.__init__(
            self,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs
        )
        print(
            "Initialized ATMTokenizerFast using the following vocab: {} "
            "and special tokens: mask_token:{}, pad_token:{}, bos_token:{}, eos_token:{}, "
            "unk_token:{}, sep_token:{}, cls_token:{}, create_segment_ids:{}".format(vocab_file,
                                                              mask_token, pad_token,
                                                              bos_token, eos_token, unk_token, sep_token, cls_token, create_segment_ids))

    @staticmethod
    def create_fast_tokenizer(slow_tokenizer, create_token_type_id):
        return ATMConverter(slow_tokenizer, create_type_ids=create_token_type_id).converted()

    # pylint: disable=signature-differs
    def save_vocabulary(self, save_directory, filename_prefix):
        if not os.path.isdir(save_directory):
            print(f"ERROR: Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
            print(f"Copy vocab file to {out_vocab_file}")

        return (out_vocab_file,)
    # pylint: enable=signature-differs

class MT5TokenizerFastWithMask(MT5TokenizerFast):
    def __init__(
            self,
            **kwargs):
        super().__init__(**kwargs)
        self.mask_token = self.additional_special_tokens[-1]
