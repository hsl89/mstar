from transformers import T5Tokenizer
from typing import Optional, Dict, Any, List

class SentencepieceTokenizer(T5Tokenizer):
    def __init__(
        self,
        vocab_file,
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        extra_ids=0,
        additional_special_tokens=None,
        sample_subwords=False,
        sampling_size=64,
        alpha=0.1,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        super().__init__(
            vocab_file,
            eos_token,
            unk_token,
            pad_token,
            extra_ids,
            additional_special_tokens,
            sp_model_kwargs,
            **kwargs,
        )

        # Set subword_sampling parameters here since we need to preserve the signature of `_tokenize`
        self.sample_subwords = sample_subwords
        self.sampling_size = sampling_size
        self.alpha = alpha

    def _tokenize(self, text: str) -> List[str]:
        """Take as input a string and return a list of strings (tokens) for words/sub-words. Subword sampling is used with the parameters specified during initiatlization if enabled."""
        return self.sp_model.encode(
            text, 
            out_type=str, 
            enable_sampling=self.sample_subwords, 
            alpha=self.alpha, 
            nbest_size=self.sampling_size
            )