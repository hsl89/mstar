from mstar.models.gpt2 import (
    MStarGPT2Config,
    MStarGPT2LMHeadConfig,
    MStarGPT2Model,
    MStarGPT2LMHeadModel,
)
from mstar.models.t5 import MStarT5Config, MStarT5ForConditionalGeneration
from mstar.models.atm_models import (
    PreLnConfig,
    PreLnForMaskedLMConfig,
    PreLnForMaskedLM,
    PreLnModel,
    PreLNSeq2SeqConfig,
    PreLNSeq2SeqForConditionalGeneration,
    ATMSeq2SeqConfig,
    ATMSeq2SeqForConditionalGeneration,
    MT5EncoderConfig,
    MT5,
    MT5ForMaskedLMConfig,
    MT5ForMaskedLM,
)
from mstar.tokenizers.gptbart import GPT2BartTokenizer
from mstar.tokenizers.sentencepiece import SentencepieceTokenizer
from mstar.tokenizers.atm_tokenizers import ATMTokenizerFast, MT5TokenizerFastWithMask
from transformers import GPT2Tokenizer, T5Tokenizer

config_dict = {
    "mstar-gpt2": MStarGPT2Config,
    "mstar-gpt2LMHead": MStarGPT2LMHeadConfig,
    "atm-PreLnForMaskedLM": PreLnForMaskedLMConfig,
    "atm-PreLn": PreLnConfig,
    "atm-PreLNSeq2Seq": PreLNSeq2SeqConfig,
    "atm-Seq2Seq": ATMSeq2SeqConfig,
    "atm-MT5": MT5EncoderConfig,
    "atm-MT5ForMaskedLM": MT5ForMaskedLMConfig,
    "mstar-t5": MStarT5Config,
}

model_class_dict = {
    "mstar-gpt2": MStarGPT2Model,
    "mstar-gpt2LMHead": MStarGPT2LMHeadModel,
    "atm-PreLnForMaskedLM": PreLnForMaskedLM,
    "atm-PreLn": PreLnModel,
    "atm-PreLNSeq2Seq": PreLNSeq2SeqForConditionalGeneration,
    "atm-Seq2Seq": ATMSeq2SeqForConditionalGeneration,
    "atm-MT5": MT5,
    "atm-MT5ForMaskedLM": MT5ForMaskedLM,
    "mstar-t5": MStarT5ForConditionalGeneration,
}

tokenizer_class_dict = {
    "mstar-gpt2": [GPT2BartTokenizer, SentencepieceTokenizer],
    "mstar-gpt2LMHead": [GPT2BartTokenizer, SentencepieceTokenizer],
    "atm-PreLnForMaskedLM": [ATMTokenizerFast],
    "atm-PreLn": [ATMTokenizerFast],
    "atm-PreLNSeq2Seq": [ATMTokenizerFast],
    "atm-Seq2Seq": [ATMTokenizerFast],
    "atm-MT5": [MT5TokenizerFastWithMask],
    "atm-MT5ForMaskedLM": [MT5TokenizerFastWithMask],
    "mstar-t5": [T5Tokenizer, SentencepieceTokenizer],
}


tokenizer_mapping = {
    "GPT2BartTokenizer": GPT2BartTokenizer,
    "ATMTokenizerFast": ATMTokenizerFast,
    "MT5TokenizerFastWithMask": MT5TokenizerFastWithMask,
    "GPT2Tokenizer": GPT2Tokenizer,
    "T5Tokenizer": T5Tokenizer,
    "SentencepieceTokenizer": SentencepieceTokenizer,
}
