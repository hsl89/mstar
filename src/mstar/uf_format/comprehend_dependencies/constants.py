START = "start"
END = "end"
DOCUMENT_ID = "document_id"
SENT_ID = "sent_id"
TOKEN_INDEX = "token_index"
TOKENS = "tokens"
EXTENT = "extent"
TOKEN_SENT_INDEX = "token_sent_index"
SENTENCE_INDEX = "sentence_index"
METADATA = "metadata"
RAW_TEXT = "raw_text"
BYPASS = "bypass"
# sentiment related fields
DOCUMENT_LABEL = "document_label"
SENTENCES_LABEL = "sentences_label"

ENTITIES_LABEL = "entities_label"
TAG = "tag"
COREFERENCE_LABEL = "coreference_label"

# relations related field
RELATION_LABEL = "document_relation_label"

# events extraction related field
EVENT_ID = "event_id"
ANNOTATIONS = "annotations"
EVENTS = "events"
TRIGGER = "trigger"
EVENT_TYPE = "event_type"
EVENT_MENTION_ID = "event_mention_id"
ARG_ENTITIES = "arg_entities"
ENTITY_MENTION_ID = "entity_mention_id"
ENTITY_TYPE_IDS = "entity_type_ids"
ENTITY_ID = "entity_id"
NONE_ENTITY = "NONE"
ENTITY_TYPE = "entity_type"
ENTITY_SUB_TYPE = "entity_subtype"
TRIGGERS = "triggers"
ARGUMENTS = "arguments"
NAME = "name"
VALUES = "values"
ARG_TYPE = "arg_type"
EVENT_ARGUMENTS = "event_arguments"
EVENT_TRIGGERS = "event_triggers"
SCORE = "score"
PREDICTIONS = "predictions"

# events coreference related  fields
EVENT_TRIGGERS_GROUPS = "event_triggers_groups"
EVENT_ARGUMENTS_GROUPS = "event_arguments_groups"
MENTIONS = 'mentions'
GROUPS = 'groups'
COREFERENCE = 'coreference'

MISSING_DATA_PLACEHOLDER = "<MISSING>"

#custom sync ner model dir
CUSTOM_SYNC_SAGEMAKER_ENDPOINT_MODEL_DIR = "/opt/ml/model"

ENVIRONMENT_VARIABLES = "environment_variables"

PER_TOKEN_LENGTH_LIST = "per_token_length_list"
PER_SUBTOKEN_LENGTH_LIST = "per_subtoken_length_list"
