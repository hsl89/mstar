# Task Embedding Model

## Index of model files:
1. model.py - Base task embedding model. It is a multihead model (see corresponding directory) that adds the learned task embedding token at the start of the sequence and allows the encoder to attend to this as well. The handling of the task embedding is done in the encoder which is a modified XLM-R encoder from xlmr.py
2. xlmr.py - Modified encoder file. This file is modified from the RobertaModel HF code. The only difference is an added task embedding layer. In addition to the standard Roberta input, the model accepts a tensor task\_ids which is mapped to a task embedding. This is concatenated to the word embedding output before passing it to the encoder. 
3. target\_task\_model.py - Same as model.py but meant for target task fine-tuning and benchmarking against single task models. Modified to be used with the standard HF trainer. This model initializes a new task embedding for the task, but you can manually load one too.
4. target\_task\_mixture\_model.py - When performing target task fine-tuning this model learns a mixture of the pre-finetuning task embeddings instead of initializing a new one. This is handled in mixture\_xlmr.py
5. mixture\_xlmr.py - Modified encoder file. This file is modified from the RobertaModel HF code. It has an attribute to load the pre-finetuned task embeddings and another to learn the weights at which to mix them.
