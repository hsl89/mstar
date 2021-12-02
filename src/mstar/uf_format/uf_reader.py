import logging
from enum import Enum
from typing import Iterable
 
from .comprehend_dependencies.document import Document
# from comprehend.model.commons.data.format.base_data_format import BaseDataFormat
from .comprehend_dependencies.uf_format import UFDataFormat, CHARACTER_LEVEL, TOKEN_LEVEL
from .comprehend_dependencies.token_span import TokenSpan
# from comprehend.model.commons.utils import config_logging
 
# from comprehend.model.sequencetagging.data  .constants import LABEL_FORMAT, ENTITIES_LABEL
LABEL_FORMAT = "{}-{}"
ENTITIES_LABEL = "entities_label"
"""
 UF, short for Universal Format, is comprehend internal data format for storing annotations
 for various NLP tasks.
 
 A sample UF document for NER looks like below:
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
 
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
 
 
class SeqTagTask(Enum):
    ALL = 0  # all tasks
    NER = 1  # named_entity
 
 
class SeqTagDataReader(UFDataFormat):
    '''
    Data reader for sequence tagging task. Now only support read data from universal format.
    '''
 
    def __init__(self,
                 tasks: Iterable[SeqTagTask] = (SeqTagTask.ALL,),
                 encoding='iob',
                 overlap_ratio=0.5,
                 exact_match=False,
                 span_key="named_entity",
                 span_features=None,
                 **kwargs):
        if exact_match:
            logger.warning('overlap_ratio will be ignored since exact match is set to True!')
        # by default extract annotations for all tasks
        self.tasks = tasks
        # encoding scheme for parsing data.
        self.encoding = encoding
        self.overlap_ratio = overlap_ratio
        self.exact_match = exact_match
        self.span_key = span_key
        self.span_features = span_features
        self.span_label = ENTITIES_LABEL if span_key == "named_entity" else span_key.upper()


        # by default fails on any kind of failures
        self.skip_failure = kwargs.get("skip_failure", False)

    @staticmethod
    def parse_named_entity_annotations(uf_json,
                                       document: Document,
                                       encoding,
                                       overlap_ratio,
                                       exact_match,
                                       span_key,
                                       span_features,
                                       span_label):
        '''
        Keep this method static in order other task(like RE Coref) can reuse this logic
        '''
        level = UFDataFormat.get_segmentation_level(uf_json)
        if level == TOKEN_LEVEL:
            # here since we already have token level data, which means tags are already contained,
            # then don't need to parse. This will happen usually in pre-tokenized task like CoNLL 2003 NER
            return

        elif level == CHARACTER_LEVEL:
            # here we need to assign tag for each token from its annotation
            try:
                named_entity_annotations = uf_json['annotations'][span_key]
            except Exception as e:
                # if named entity annotations are not present, raise exception
                raise ValueError("UF document {0} doesn't contain {1} annotations".format(
                                 document.metadata["document_id"], span_key))

            if named_entity_annotations:
                token_spans = document.create_span_by_char_offsets_batch(named_entity_annotations,
                                                                         overlap_ratio,
                                                                         exact_match)
            else:
                token_spans = []


            span_features_inferred = set()
            for token_span in token_spans:
                # now need to assign iob/iobes encoded labels for each token 
                span_features_inferred = span_features_inferred.union(SeqTagDataReader.assign_encoded_labels(token_span,
                                                                                                             encoding,
                                                                                                             span_features))

            if span_features is None:
                span_features = span_features_inferred

            # now need to assign labels "O" to tokens that don't have any tags
            for token in document.tokens:
                for span_feature in span_features:
                    if not token.get(span_feature):
                        token.set(span_feature, 'O')

            document.set(span_label, token_spans)

    @staticmethod
    def assign_encoded_labels(token_span: TokenSpan, encoding, span_features=None):
        '''
        Given a token span, it will assign tag for each token based on encoding scheme
        '''
        if span_features is None:
            span_features = set(token_span.attributes.keys()).difference({"start", "end", "extent"})
        for span_feature in span_features:
            feature_value = token_span.get(span_feature)
            if feature_value is None:
                continue

            if encoding == 'iob':
                token_span.start_token.set(span_feature, LABEL_FORMAT.format('B', feature_value))
                for token in token_span.tokens[1:]:
                    token.set(span_feature, LABEL_FORMAT.format('I', feature_value))

            elif encoding == 'iobes':
                if len(token_span.tokens) == 1:
                    token_span.start_token.set(span_feature, LABEL_FORMAT.format('S', feature_value))

                else:
                    token_span.start_token.set(span_feature, LABEL_FORMAT.format('B', feature_value))
                    token_span.end_token.set(span_feature, LABEL_FORMAT.format('E', feature_value))
                    for token in token_span.tokens[1:-1]:
                        token.set(span_feature, LABEL_FORMAT.format('I', feature_value))
 
            else:
                raise ValueError('Currently only iob and iobes encoding are supported but %s provided!'.format(encoding))
        return span_features
 
    def read_json(self, uf_json, infer_mode=False):
        d = UFDataFormat.create_base_document(uf_json)
        if SeqTagTask.NER in self.tasks or SeqTagTask.ALL in self.tasks:
            self.parse_named_entity_annotations(uf_json, d,
                                                encoding=self.encoding,
                                                overlap_ratio=self.overlap_ratio,
                                                exact_match=self.exact_match,
                                                span_key=self.span_key,
                                                span_features=self.span_features,
                                                span_label=self.span_label)

        return d
