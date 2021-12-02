from typing import Dict

from .attributes_base import AttributesBase
from .constants import START, END


class Token(AttributesBase):
    """
    A ``Token`` is a generic class that contains piece text and the attributes
    associate with it. Text should only be passed in at initialization
    """

    def __init__(self,
                 text: str,
                 attributes: Dict[str, object] = None):
        super().__init__(attributes)
        self._text = text
        self._validate_token_span()

    def _validate_token_span(self):
        if self.contains_start_end_offsets:
            # sanity check to prevent illegal span from being created
            start = self.get(START)
            end = self.get(END)
            if (end - start) != len(self._text):
                raise ValueError("Illegal Token %s : end - start"
                                 ":= %d doesn't match len(text) := %d" % (self.text,
                                                                          end - start,
                                                                          len(self.text)))

    @property
    def text(self):
        """
        Make Token.text read-only, avoid accidental modification
        """
        return self._text

    @property
    def char_array(self):
        """
        Return the character representation of the underlying text
        """
        return list(self._text)

    @property
    def contains_start_end_offsets(self):
        """
        Check whether the token comes with character offsets or not
        """
        return (self.get(START) is not None) and \
               (self.get(END) is not None)

    def __str__(self) -> str:
        return "{{Token: text='{0}'," \
               " token_attributes={1}}}".format(self._text,
                                                self._attributes)


class SubwordToken(Token):
    """
    Some subword tokenizers when splitting words add two hashes (##) as prefix to the token. This class represents those tokens
    and ensures the sanity of original start and end offsets by taking into consideration that two extra hashes
    """
    BERT_TOKENIZER_SUBWORD_PREFIX = "##"

    def __init__(self,
                 text: str,
                 attributes: Dict[str, object] = None, subword_prefix: str = BERT_TOKENIZER_SUBWORD_PREFIX):
        """
        :param text: raw text
        :param attributes: attributes associated with token , for example start and end offsets in the document
        :param subword_prefix: the prefix tokenizer uses to append to tokens if they were a subword
        """
        self._subword_prefix = subword_prefix
        super().__init__(text, attributes)

    def _validate_token_span(self):
        if self.text is None or self.text == '':
            raise ValueError("Token with empty string not allowed")
        if self.text[:len(self._subword_prefix)] != self._subword_prefix:
            # Not actually a subword . Peform the same validations as Token
            super()._validate_token_span()
            return

        if self.contains_start_end_offsets:
            # sanity check to prevent illegal span from being created
            start = self.get(START)
            end = self.get(END)
            if (end - start) != len(self._text) - len(self._subword_prefix):
                raise ValueError("Illegal Token %s : end - start"
                                 ":= %d doesn't match len(text) - len(prefix_string) := %d" % (self.text,
                                                                                               end - start,
                                                                                               len(self.text) - len(self._subword_prefix)))

    def get_prefix_stripped_string(self):
        if self.text[:len(self._subword_prefix)] == self._subword_prefix:
            return self.text[len(self._subword_prefix):]
        else:
            return self.text

    @property
    def subword_prefix(self):
        return self._subword_prefix
