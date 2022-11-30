from typing import List
from tokenizers.pre_tokenizers import BertPreTokenizer
import string

class Tokenizer:
    PROTECTED_CHARS = [chr(9777 + i) for i in range(len(string.whitespace))]
    WS_PROTECTED_TUPLES = [(ws, pc, ws + pc + ws) for ws, pc in zip(string.whitespace, PROTECTED_CHARS)]

    def __init__(self) -> None:
        self.tokenizer = BertPreTokenizer()

    def _protect_whitespace(self, text: str) -> str:
        result = text
        
        for whitespace_char, _, protected_char_w_whitespace in Tokenizer.WS_PROTECTED_TUPLES:
            result = result.replace(whitespace_char, protected_char_w_whitespace)

        return result

    def _restore_whitespace(self, text: str) -> str:
        result = text
        
        for whitespace_char, protected_char, _ in Tokenizer.WS_PROTECTED_TUPLES:
            result = result.replace(protected_char, whitespace_char)

        return result

    def tokenize(self, text: str) -> List[str]:
        protected_str = self._protect_whitespace(text)
        tokenized = self.tokenizer.pre_tokenize_str(protected_str)
        tokens, _ = list(zip(*tokenized))

        return list(tokens)

    def detokenize(self, tokens: List[str]) -> str:
        protected_text = ''.join(tokens)
        return self._restore_whitespace(protected_text)
        