import os
import copy
import json
from shutil import copyfile
from transformers import MT5TokenizerFast
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast, PreTrainedTokenizerBase
from transformers.convert_slow_tokenizer import SpmConverter
from tokenizers import processors, AddedToken
import sentencepiece as spm

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_file": "tokenizer.json"}

SPECIAL_TOKENS_MAP_FILE = "special_tokens_map.json"
TOKENIZER_CONFIG_FILE = "tokenizer_config.json"
TOKENIZER_FILE = "tokenizer.json"

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
    def save_vocabulary(self, save_directory, filename_prefix=None):
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
    

    def save_pretrained(self, save_directory, filename_prefix=None):
        if os.path.isfile(save_directory):
            print(f"Provided path ({save_directory}) should be a directory, not a file")
            return
        
        os.makedirs(save_directory, exist_ok=True)
        special_tokens_map_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + SPECIAL_TOKENS_MAP_FILE
        )
        tokenizer_config_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + TOKENIZER_CONFIG_FILE
        )

        tokenizer_config = copy.deepcopy(self.init_kwargs)
        if len(self.init_inputs) > 0:
            tokenizer_config["init_inputs"] = copy.deepcopy(self.init_inputs)
        for file_id in self.vocab_files_names:
            tokenizer_config.pop(file_id, None)

        # Sanitize AddedTokens
        def convert_added_tokens(obj, add_type_field=True):
            if isinstance(obj, AddedToken):
                out = obj.__getstate__()
                if add_type_field:
                    out["__type"] = "AddedToken"
                return out
            elif isinstance(obj, (list, tuple)):
                return list(convert_added_tokens(o, add_type_field=add_type_field) for o in obj)
            elif isinstance(obj, dict):
                return {k: convert_added_tokens(v, add_type_field=add_type_field) for k, v in obj.items()}
            return obj
        
        tokenizer_config = convert_added_tokens(tokenizer_config, add_type_field=True)
        tokenizer_class = self.__class__.__name__
        tokenizer_config["tokenizer_class"] = tokenizer_class

        with open(tokenizer_config_file, "w", encoding="utf-8") as f:
            out_str = json.dumps(tokenizer_config, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            f.write(out_str)
        print(f"tokenizer config file saved in {tokenizer_config_file}")

        write_dict = convert_added_tokens(self.special_tokens_map_extended, add_type_field=False)

        with open(special_tokens_map_file, "w", encoding="utf-8") as f:
            out_str = json.dumps(write_dict, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            f.write(out_str)
        print(f"Special tokens file saved in {special_tokens_map_file}")

        file_names = (tokenizer_config_file, special_tokens_map_file)

        tokenizer_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + TOKENIZER_FILE
        )
        self.backend_tokenizer.save(tokenizer_file)
        file_names = file_names + (tokenizer_file,)

        vocab_files = self.save_vocabulary(save_directory, filename_prefix=filename_prefix)

        return file_names + vocab_files
    # pylint: enable=signature-differs

class MT5TokenizerFastWithMask(MT5TokenizerFast):
    def __init__(
            self,
            **kwargs):
        super().__init__(**kwargs)
        self.mask_token = self.additional_special_tokens[-1]
