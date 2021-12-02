import logging
from builtins import str
from typing import List, Dict, Union, Optional
from uuid import uuid4

from .attributes_base import AttributesBase
from .constants import DOCUMENT_ID, TOKEN_INDEX, TOKEN_SENT_INDEX, SENTENCE_INDEX, START, \
    END, PER_TOKEN_LENGTH_LIST, PER_SUBTOKEN_LENGTH_LIST
from .token import Token
from .token_span import TokenSpan

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Document(AttributesBase):
    """
    Document is the container class to operate upon a list of tokens
    """

    def __init__(self,
                 sentences: Union[List[List[Union[Token, str]]], List[Union[Token, str]], List[TokenSpan]],
                 metadata: Dict[str, object] = None,
                 attributes: Dict[str, object] = None):
        super().__init__(attributes)

        if metadata is not None:
            self.metadata = metadata
        else:
            self.metadata = dict()

        if DOCUMENT_ID not in self.metadata:
            random_generated_id = str(uuid4())
            logger.debug("Document is initialized with no document_id, assign uuid %s as document_id",
                         random_generated_id)
            self.metadata[DOCUMENT_ID] = random_generated_id

        if PER_TOKEN_LENGTH_LIST not in self.metadata:
            self.metadata[PER_TOKEN_LENGTH_LIST] = list()

        if PER_SUBTOKEN_LENGTH_LIST not in self.metadata:
            self.metadata[PER_SUBTOKEN_LENGTH_LIST] = list()

        # if we received a list of tokens, wrap it in a list
        sentences = sentences[:] if len(sentences) > 0 and (isinstance(sentences[0], list) or isinstance(sentences[0], TokenSpan)) \
            else [sentences, ]

        # filter out empty sentences
        sentences = [st for st in sentences if len(st) > 0]
        if len(sentences) > 0 and isinstance(sentences[0], list) and isinstance(sentences[0][0], str):
            # convert token in sentences to Token class, if given str
            sentences = Document.convert_str_to_tokens(sentences)

        self._sentences = self._create_sentence_from_token_spans(sentences)

        self._tokens = self._get_all_tokens(self._sentences)

        self.with_char_offsets = self._is_char_offsets_present(self._tokens)

    def _get_all_tokens(self, sentences):
        tokens = list()
        doc_token_idx = 0
        for si, sent in enumerate(sentences):
            for token in sent:
                # set indices for each token in the document
                token.set(TOKEN_INDEX, doc_token_idx)
                token.set(TOKEN_SENT_INDEX, si)
                tokens.append(token)
                doc_token_idx += 1
        return tokens

    def _is_char_offsets_present(self, tokens):
        # check whether char offsets are provided or not
        # this will be useful in deciding whether we can
        # provide character level offsets addressing or not

        with_char_offsets = True

        for token in tokens:
            if not token.contains_start_end_offsets:
                with_char_offsets = False
                break

        return with_char_offsets

    def _create_sentence_from_token_spans(self, sentences):
        sent_idx = 0
        _sentences = list()
        for sent in sentences:
            # sentence is just an instance of token span
            # with sentence index
            if not isinstance(sent, TokenSpan):
                sent_span = TokenSpan(tokens=sent)
            else:
                sent_span = sent

            sent_span.set(SENTENCE_INDEX, sent_idx)
            sent_idx += 1
            _sentences.append(sent_span)
        return _sentences

    @property
    def tokens(self) -> List[Token]:
        """
        Make Document.tokens read-only, avoid accidental modification
        """
        return self._tokens

    @property
    def sentences(self) -> List[TokenSpan]:
        """
        Make Document.segments read-only, avoid accidental modification
        """
        return self._sentences

    @property
    def num_tokens(self) -> int:
        return len(self._tokens)

    @property
    def num_sentences(self) -> int:
        return len(self._sentences)

    def create_span_by_token_indices(self,
                                     start: int,
                                     end: int) -> Optional[TokenSpan]:
        """
        Create a 'view' over the self._tokens list to represent a span
        given start and end token indices. The start are inclusive, while
        end are exclusive.

        Equal to return self._tokens[start: end]
        """
        if start <= end:
            tokens = self.tokens[start: end]
            return TokenSpan(tokens)

    def create_span_by_char_offsets(self, start: int, end: int, overlap_ratio: float = 0, exact_match: bool = False) -> Optional[TokenSpan]:
        """
        Implementation of create_span_by_char_offsets that uses the batch logic internally
        """
        result = self.create_span_by_char_offsets_batch([{"start": start, "end": end}], overlap_ratio=overlap_ratio,
                                                        exact_match=exact_match)
        if len(result) == 0:
            return None
        return result[0]

    def create_span_by_char_offsets_batch(self, entities, overlap_ratio: float = 0, exact_match: bool = False):
        """
        Create a list of TokenSpan over the tokens in this document based on entities given

        This replaces the old create_span_by_char_offsets, however instead of doing one pass
        over the document for each entity, we do one pass and collect all TokenSpans with matching
        entities, making it more time-efficient.
        """
        if overlap_ratio < 0 or overlap_ratio > 1:
            raise ValueError("overlap ratio should be >= 0 and <= 1, currently value: %f" % overlap_ratio)

        if not self.with_char_offsets:
            raise ValueError("Trying to locate span when no character level information are provided")
        sorted_entities = sorted(entities, key=lambda e: e["start"])
        cur_token_index = 0
        tokens = self.tokens
        token_spans = []
        for entity in sorted_entities:
            span_token_start = None
            span_token_end = None

            while cur_token_index < len(tokens) and tokens[cur_token_index].get(START) < entity["end"]:
                if tokens[cur_token_index].get(END) > entity["start"]:
                    token_len = tokens[cur_token_index].get(END) - tokens[cur_token_index].get(START)
                    allignment_len = min(tokens[cur_token_index].get(END), entity["end"]) - max(tokens[cur_token_index].get(START),
                                                                                                entity["start"])
                    if float(allignment_len) / token_len >= overlap_ratio:
                        if span_token_start is None:
                            span_token_start = span_token_end = cur_token_index
                        else:
                            span_token_end = cur_token_index

                cur_token_index += 1

            if span_token_start is not None:
                span_tokens = tokens[span_token_start: span_token_end + 1]
                token_span = TokenSpan(tokens=span_tokens)
                for k, v in entity.items():
                    if k in {"start", "end", "extent"}:
                        continue
                    token_span.set(k, v)
                cur_token_index = span_token_start
                if not exact_match or (
                        token_span.start_token.get(START) == entity["start"] and token_span.start_token.get(END) == entity["end"]):
                    token_spans.append(token_span)

        return token_spans

    def __str__(self) -> str:
        str_sentences = ", ".join([str(sent) for sent in self.sentences])
        return "{{Document: sentences=[{0}], " \
               "document_attributes={1}, metadata={2}}}".format(str_sentences, self._attributes, self.metadata)

    @staticmethod
    def convert_str_to_tokens(sents):
        """
        If given strs in sentence, convert them into Token first
        """
        _sentences = list()
        for s1 in sents:
            s2 = [Token(ts) for ts in s1]
            _sentences.append(s2)
        return _sentences
