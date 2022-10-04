# M5PretrainTokenizers

This folder includes pretrained tokenizers' code imported from [M5Tokenizers](https://code.amazon.com/packages/M5Tokenizers/blobs/mainline/--/src/m5_tokenizers/tokenizers/m5_sentencepiece_tokenizer.py) with commit hash 57c85875a5524a6dc9a47b0ac130c78931748161

## Use mstar AutoTokenizer to load M5SentencepieceTokenizer
List of available tokenizer ids:
- m5-Bert-5B-20220913
- m5-Bert-50B-20220913

```python
from mstar import AutoTokenizer

## Special_tokens are required when loading M5SentencepieceTokenizer
## This can make sure the embeddings are correct. 
special_tokens = [
    "[MASK]","[SEP]","[CLS]","[en_CA]","[mr_IN]","[pl_PL]","[sv_SE]","[en_US]","[it_IT]","[ta_IN]",
    "[te_IN]","[ml_IN]","[nl_NL]","[en_AU]","[en_IN]","[hi_IN]","[ko_KR]","[de_DE]","[pt_PT]","[ja_JP]",
    "[zh_CN]","[pt_BR]","[en_AE]","[zh_TW]","[fr_FR]","[kn_IN]","[es_MX]","[ar_AE]","[cs_CZ]","[en_GB]",
    "[en_SG]","[he_IL]","[tr_TR]","[ru_RU]","[es_ES]"
]

## For model_id to tokenizer_class mapping, please refer to model_factory.py
model = AutoTokenizer.from_pretrained('m5-Bert-5B-20220913', additional_special_tokens=special_tokens)

```
