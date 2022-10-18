MODEL_INPUT='/mnt/colehawk/bedrock_prod_automodels/stage_2/1_9B/alexatm/'
MODEL_NAME='mstar-t5-1-9B-bedrock'
REVISION='stage_2_alexatm'
TOKENIZER_INPUT='/mnt/colehawk/bedrock_prod_automodels/tokenizer/'

aws s3 sync $TOKENIZER_INPUT s3://mstar-models/tokenizers/$MODEL_NAME/$REVISION
aws s3 sync $MODEL_INPUT s3://mstar-models/models/$MODEL_NAME/$REVISION
