# Multihead Model

Model with one shared encoder (XLMR) and one decoder for each dataset. 
The datasets to be used are specified in a JSON format checked in to uf\_format/large\_args which gives information about the label vocab, location of the train and validation files.

## Details:
1. We scale the loss by the number of labels as specified by Muppet. 
2. The difference between model.py and target\_task\_model.py is that the former is used for pre-finetuning with multiple datasets but needs to be used with an MTL trainer to handle the same, but the latter is for target task finetuning on a single task with the standard HF Trainer and benchmarking against single task models.
3. When loading a model for target task finetuning, if the target task is seen during pre-finetuning you can load that particular decoder head as well. If it is an unseen task then you initialize a new head.
