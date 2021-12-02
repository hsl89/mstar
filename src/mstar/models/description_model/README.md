# Description Models

## Index of model files
1. model.py - Base Multihead Description Model - Multihead model (see directory) that has a separate description encoder. The data loader is meant to provide additional input which is description\_input\_ids and description\_attention\_mask. This model obtains the separately encoded output of the input sequence and description and concatenates both before passing it to the classifier. The encoders used are standard XLM-R models
2. target\_task\_model.py - Same as model.py but modified for target task fine tuning to run on the standard HF trainer
3. sequence\_model.py - Multihead model with a separate description encoder. But the output of the description encoder is fed as an additional token at the start of the input sequence (similar to the task embedding model). The intention is that the sequence encoder can learn to attend to this token during pre-finetuning. The encoder used is a modified XLM-R model from xlmr.py 
4. xlmr.pt - Modified RobertaModel HF code. Accepts the description encoder output in addition to standard Roberta input. Adds the description output as the first element of the sequence and allows the encoder to attend to it. 
5. target\_task\_sequence\_model.py - Same as sequence\_model.py but modified for target task fine-tuning
6. interpolated\_model.py - Description model but instead of adding the output of the encoder  
