# ATMPretrainModels

This folder includes pretrained models' code imported from [AlexaTeacherModelExperiments](https://code.amazon.com/packages/AlexaTeacherModelExperiments/trees/mainline) with commit hash dc015a29be1c23b2e46e2d54bacd40dc466f4c2e

## Use mstar AutoModel to load ATM pretrained models
List of model names that can be loaded:
- atm-PreLnForMaskedLM-10B
- atm-PreLnForMaskedLM-2B
- atm-PreLn-10B
- atm-PreLn-2B
- atm-PreLNSeq2Seq-20B
- atm-PreLNSeq2Seq-5B
- atm-Seq2Seq-406M
- atm-MT5-1B
- atm-MT5ForMaskedLM-1B

```python
from mstar import AutoModel

## For model_id to model_class mapping, please refer to model_factory.py
model = AutoModel.from_pretrained('atm-PreLnForMaskedLM-10B')

```
