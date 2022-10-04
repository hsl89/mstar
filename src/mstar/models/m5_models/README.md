# M5PretrainModels

This folder includes pretrained models' code imported from [M5Models](https://code.amazon.com/packages/M5Models/trees/mainline) with commit hash 6a9b3e5c83c2f42b34d905b6195f8c0e07e13544

## Use mstar AutoModel to load M5BertForPreTrainingPreLN
List of available model ids that can be loaded:
- m5-Bert-5B-20220913
- m5-Bert-50B-20220913

```python
from mstar import AutoModel

## For model_id to model_class mapping, please refer to model_factory.py
model = AutoModel.from_pretrained('m5-Bert-5B-20220913')

```
