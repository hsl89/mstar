from typing import List, Dict

from .attributes_base import AttributesBase
from .token import Token


class TokenSpan(AttributesBase):
    """
    A ``TokenSpan`` is a generic class that presents a sequence of tokens.

    It covers the use case as both a sentence or several consecutive words
    as presented in the document
    """

    def __init__(self,
                 tokens: List[Token],
                 attributes: Dict[str, object] = None):
        super().__init__(attributes)
        self._tokens = tokens

    def __len__(self):
        return len(self._tokens)

    def __iter__(self):
        for token in self._tokens:
            yield token
        # raise StopIteration

    @property
    def tokens(self) -> List[Token]:
        """
        Make Segment.tokens read-only, avoid accidental modification
        """
        return self._tokens

    @property
    def start_token(self) -> Token:
        """
        Return the start token index of this segment in the document
        """
        return self._tokens[0]

    @property
    def end_token(self) -> Token:
        """
        Return the end token index of this segment in the document
        """
        return self._tokens[-1]

    @property
    def text(self) -> List[str]:
        return [t.text for t in self.tokens]

    def __str__(self) -> str:
        str_tokens = ", ".join([str(t) for t in self._tokens])
        return "{{TokenSpan: tokens=[{0}], " \
               "span_attributes={1}}}".format(str_tokens,
                                              self._attributes)
