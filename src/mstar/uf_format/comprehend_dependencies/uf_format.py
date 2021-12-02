import logging
import ujson
from enum import Enum
from typing import Iterable

from .constants import START, END, DOCUMENT_LABEL, SENTENCES_LABEL, RELATION_LABEL, TOKEN_INDEX, ENTITIES_LABEL, COREFERENCE_LABEL, RAW_TEXT
from .document import Document
from .base_data_format import BaseDataFormat
from .token import Token

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

"""
UF, short for Universal Format, is comprehend internal data format for storing annotations
for various NLP tasks.

A sample UF document looks like below:
{
  "annotations": {
    "named_entity": [
      {
        "start": ...,
        "end": ...,
        "tag": "...",
        "extent": "...",
        "<attrs>": "...",
        ...
      }
    ],
    ...[annotations for other tasks]...
    "sentiment": {
      "sentences": [...],
      "document": "LABEL"
    },
  },
  "raw_text": "...",
  "metadata": {
    "document_id": ".....",
  },
  "preprocessing": {
    "segments": {
      "level": "character|token"
      "sentences": [
        {
          "start": ...,
          "end": ...,
        },
      ]
      "tokens": [
        {
          "start": ...,
          "end": ...,
          "<attrs>": ...
        }
      ]
    }
  }
}

"""


class Task(Enum):
    ALL = 0  # all tasks
    NER = 1  # named_entity
    SENTIMENT = 2  # sentiment
    KPE = 3  # key_phrase
    COREF = 4  # coreference
    RE = 5  # relations, or relation extraction


CHARACTER_LEVEL = "character"
TOKEN_LEVEL = "token"


@BaseDataFormat.register("UF")
class UFDataFormat(BaseDataFormat):
    def __init__(self, tasks: Iterable[Task] = (Task.ALL,), **kwargs):
        # by default extract annotations for all tasks
        self.tasks = tasks
        # IOB format: B-PER, I-PER, O, O
        self.to_iob = kwargs.get("IOB", False)
        # IOBES format: S-PER, B-PER, I-PER, E-PER, O
        self.to_iobes = kwargs.get("IOBES", False)
        # skip malformed documents
        # by default fails on any kind of failures
        self.skip_failure = kwargs.get("skip_failure", False)
        # minimal overlap ratio
        # the minimal ratio requires for a token to be assigned to a TokenSpan
        self.overlap_ratio = kwargs.get("overlap_ratio", 0)
        # exact match
        # setting to True will only match TokenSpans that are exact matches
        self.exact_match = kwargs.get("exact_match", False)

    @staticmethod
    def get_segmentation_level(uf_json):
        try:
            level = uf_json["preprocessing"]["segments"]["level"]
            return level
        except Exception as e:
            # by default we assume character level
            return CHARACTER_LEVEL

    @staticmethod
    def create_base_document(uf_json, include_raw_text=False) -> Document:
        """
        Create a basic Document object that contains:
        1. metadata, especially document_id
        2. tokenization
        3. sentences segmentation if offered

        Task related annotations information will be added later
        by operating on the returned document
        """
        try:
            # check that document_id is presented
            _ = uf_json["metadata"]["document_id"]
        except Exception as e:
            raise ValueError("Failed to find document_id from metadata in %s" % ujson.dumps(uf_json))

        metadata = uf_json["metadata"]
        doc_attrs = dict()
        if include_raw_text:
            doc_attrs[RAW_TEXT] = uf_json["raw_text"]

        try:
            token_segments = uf_json["preprocessing"]["segments"]["tokens"]
        except Exception as e:
            raise ValueError("Failed to obtain tokenization information from UF document")

        # pre-processing can come in 2 different levels
        level = UFDataFormat.get_segmentation_level(uf_json)

        assert level in (CHARACTER_LEVEL, TOKEN_LEVEL), \
            "Unknown segmentation level passed in %s" % level

        if level == CHARACTER_LEVEL:
            raw_text = uf_json["raw_text"]
            # construct token objects
            tokens = list()
            for ts in token_segments:
                start = ts["start"]
                end = ts["end"]
                text = raw_text[start: end]
                attributes = {k: v for k, v in ts.items() if k != "extent"}
                attributes[START] = start
                attributes[END] = end
                token = Token(text, attributes)
                tokens.append(token)

            sentence_segments = None
            try:
                sentence_segments = uf_json["preprocessing"]["segments"]["sentences"]
            except Exception:
                logger.debug("No sentences segmentation found for document %s"
                             % uf_json["metadata"]["document_id"])

            if sentence_segments is None:
                # find no sentences information, assume it is single sentence
                return Document(tokens, metadata, doc_attrs)
            else:
                # now assign tokens to sentences
                # sort all segments by starts
                sentence_segments = sorted(sentence_segments, key=lambda s: s["start"])
                offsets = [(c["start"], c["end"]) for c in sentence_segments]
                sentences = UFDataFormat.assign_token_to_sentences(tokens, offsets)
                return Document(sentences, metadata, doc_attrs)
        elif level == TOKEN_LEVEL:
            # Token level
            raw_text = uf_json["raw_text"]
            tokens = list()
            for ts in token_segments:
                text = ts["extent"]
                # everything other than extent will go into attributes
                attributes = {k: v for k, v in ts.items() if k != "extent"}
                token = Token(text, attributes)
                tokens.append(token)

            # here we just skip sentence assignment since is not needed
            return Document(tokens, metadata, doc_attrs)

    @staticmethod
    def assign_token_to_sentences(tokens, sentence_offsets):
        """
        Assign token to a list of sentences offsets

        Our alignment policy make the assumption that
        every token will be assigned to exactly one sentence

        TODO: make this a policy object for configuration?
        """
        token_idx = 0
        sent_idx = 0
        sentences = []
        cur_sentence = []
        while token_idx < len(tokens):
            # our alignment policy make the assumption that
            # every token is should be assigned to exactly one sentence
            token = tokens[token_idx]
            token_start, token_end = token.get(START), token.get(END)
            if sent_idx < len(sentence_offsets):
                sent_start, sent_end = sentence_offsets[sent_idx]
                if max(sent_start, token_start) < min(sent_end, token_end):
                    # either containing or overlaps, add into current sentence
                    cur_sentence.append(token)
                    token_idx += 1
                else:
                    # not overlapping, we have 2 cases
                    if token_start >= sent_end:
                        # token_start is bigger than current sentence end
                        # move to next sentence
                        sentences.append(cur_sentence[:])
                        cur_sentence = []
                        sent_idx += 1
                    else:
                        # the current token doesn't belong to either prev sentence
                        # nor the current one, meaning there is 'hole' in the segmentation
                        logger.warning("Token %s doesn't belong to any segment, "
                                       "assigning it to the current segment" % token)
                        cur_sentence.append(token)
                        token_idx += 1
            else:
                # there are tokens not covered by any segments at the end
                # add them to the last sentence
                logger.warning("Token %s doesn't belong to any segment, "
                               "assigning it to the last segment" % token)
                cur_sentence.append(token)
                token_idx += 1
        # add the last sentence
        sentences.append(cur_sentence[:])
        return sentences

    def parse_sentiment_annotations(self, uf_json, document: Document):
        level = UFDataFormat.get_segmentation_level(uf_json)
        if Task.SENTIMENT in self.tasks or Task.ALL in self.tasks:
            if Task.SENTIMENT in self.tasks:
                if "annotations" not in uf_json or "sentiment" not in uf_json["annotations"]:
                    # if sentiment annotations is
                    raise ValueError("UF document %s doesn't contain sentiment annotations" %
                                     document.metadata["document_id"])

            if "annotations" in uf_json:
                sentiment_annotations = uf_json["annotations"].get("sentiment")
                if sentiment_annotations:
                    document_label = sentiment_annotations["document"]
                    document.set(DOCUMENT_LABEL, document_label)
                    sentences = sentiment_annotations["sentences"]
                    token_spans = document.create_span_by_char_offsets_batch(sentences, overlap_ratio=self.overlap_ratio, exact_match=self.exact_match)
                    document.set(SENTENCES_LABEL, token_spans)

    def parse_named_entity_annotations(self, uf_json, document: Document):
        # Caution: At present, our current NER model doesn't require named entity annotations
        #          at span level. This parsing is dependent on model or task type and only
        #          being used by coref model for now.
        #          So, only do this if your model needs span level information of NER.
        if Task.NER in self.tasks or Task.ALL in self.tasks:
            if Task.NER in self.tasks:
                if "annotations" not in uf_json or "named_entity" not in uf_json["annotations"]:
                    # if named entity annotations are not present, raise exception
                    raise ValueError("UF document %s doesn't contain named entity annotations" %
                                     document.metadata["document_id"])

            if "annotations" in uf_json:
                named_entity_annotations = uf_json["annotations"].get("named_entity")
                token_spans = []
                if named_entity_annotations:
                    token_spans = document.create_span_by_char_offsets_batch(named_entity_annotations, overlap_ratio=self.overlap_ratio, exact_match=self.exact_match)
                document.set(ENTITIES_LABEL, token_spans)

    def parse_coreference_annotations(self, uf_json, document: Document):
        if Task.COREF not in self.tasks and Task.ALL not in self.tasks:
            return

        # named entity are input to coref model, so parse and add it to document
        # as we need NER at span level
        if not document.get(ENTITIES_LABEL):
            self.parse_named_entity_annotations(uf_json, document)
        if Task.COREF in self.tasks or Task.ALL in self.tasks:
            if Task.COREF in self.tasks:
                if "annotations" not in uf_json or "coreference" not in uf_json["annotations"]:
                    # if coreference annotations are not present, raise exception
                    raise ValueError("UF document %s doesn't contain coreference annotations" %
                                     document.metadata["document_id"])

            if "annotations" in uf_json:
                coref_annotations = uf_json["annotations"].get("coreference")
                if coref_annotations:
                    groups = coref_annotations["groups"]
                    token_spans = []
                    if groups:
                        for group in groups:
                            mentions = group["mentions"]
                            if mentions:
                                cluster_token_spans = document.create_span_by_char_offsets_batch(mentions, overlap_ratio=self.overlap_ratio, exact_match=self.exact_match)
                                token_spans.append(cluster_token_spans)
                    document.set(COREFERENCE_LABEL, token_spans)

    def get_start_end_token_offsets(self, arg, document: Document):
        arg_token_span = document.create_span_by_char_offsets(arg["start"], arg["end"])
        return (arg_token_span.tokens[0].get(TOKEN_INDEX),
                arg_token_span.tokens[-1].get(TOKEN_INDEX) + 1)

    def set_positional_args(self,
                            key,
                            start_token_offset,
                            end_token_offset,
                            token: Token):
        """
        This method is currently used for RE parsing.
        Given a span (token_start_index, token_end_index) this token will get
        a relative position argument in the attribute denoted by key.
        This relative position indicates where this token is located w.r.t. the
        span.
        Args:
            key: attribute name to insert in the token
            start_token_offset: span start
            end_token_offset: span end
            token: token to give relative span information to
        Returns:
            None
        """
        i = token.get(TOKEN_INDEX)
        if i < start_token_offset:
            token.set(key, i - start_token_offset)
        elif start_token_offset <= i < end_token_offset:
            token.set(key, 0)
        else:
            token.set(key, i - end_token_offset + 1)

    def parse_re_annotations(self, uf_json, document: Document, infer_mode=False):
        if Task.RE in self.tasks or Task.ALL in self.tasks:
            if not infer_mode and ("annotations" not in uf_json or
                                   "relations" not in uf_json["annotations"] or
                                   len(uf_json["annotations"]["relations"]) == 0):
                # if relations annotations is
                raise ValueError(
                    "UF document %s doesn't contain relations annotations" %
                    document.metadata["document_id"])
            elif infer_mode and ("infer_query_pairs" not in uf_json):
                raise ValueError(
                    "UF document %s needs infer_query_pairs in infer_model" %
                    document.metadata["document_id"])

            # Assumed pairwise query documents for now. So len(relations) == 1.
            # Query pair data in positional embeddings per token attribute.
            # Consider: Bring schema restriction, dist restriction and pair
            # generation from ACEditor documents to pairwise documents here.
            if not infer_mode:
                relation = uf_json["annotations"]["relations"][0]
            else:
                relation = uf_json["infer_query_pairs"][0]

            # Generate positional embeddings dynamically
            arg1 = relation["arg1"]
            arg1_start_token_offset, arg1_end_token_offset = self.get_start_end_token_offsets(arg1, document)

            arg2 = relation["arg2"]
            arg2_start_token_offset, arg2_end_token_offset = self.get_start_end_token_offsets(arg2, document)

            for token in document.tokens:
                self.set_positional_args("positional_emb_arg1",
                                         arg1_start_token_offset,
                                         arg1_end_token_offset,
                                         token)

                self.set_positional_args("positional_emb_arg2",
                                         arg2_start_token_offset,
                                         arg2_end_token_offset,
                                         token)

            # Only train mode will have the ground truth label
            if not infer_mode:
                relation_label = relation["relation"]
                document.set(RELATION_LABEL, relation_label)

    def read_json(self, uf_json, infer_mode=False):
        d = UFDataFormat.create_base_document(uf_json)
        # TODO: add the parsing logic for other tasks later
        self.parse_sentiment_annotations(uf_json, d)
        self.parse_re_annotations(uf_json, d, infer_mode)

        # for NER since tag info is already in attributes of each token,
        # then we don't need to parse the tag
        self.parse_coreference_annotations(uf_json, d)
        return d

    def read(self, file_path: str) -> Iterable[Document]:
        with BaseDataFormat.compression_aware_open(file_path) as f:
            for line in f:
                uf_json = ujson.loads(line)
                try:
                    d = self.read_json(uf_json)
                    yield d
                except Exception as e:
                    if not self.skip_failure:
                        raise e
