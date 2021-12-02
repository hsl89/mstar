import numpy as np
from transformers import HfArgumentParser
import json
import torch
import argparse
from mtl_trainer import MultitaskTrainer
from mstar.uf_format.uf_reader import SeqTagTask, SeqTagDataReader
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback
from transformers import AutoModelForTokenClassification, AutoConfig, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from datasets import load_metric, Dataset, DatasetDict, concatenate_datasets
from mstar.models.task_embedding_model.target_task_model import TaskEmbeddingMultiheadModel
from mstar.models.multihead_model.target_task_model import MultiheadModel
from torch.utils.data.dataloader import DataLoader
import torchmetrics
"""
For reference: https://github.com/huggingface/notebooks/blob/master/examples/token_classification.ipynb
"""

def tokenize_and_align_labels(dataset, tokenizer):
    """
    Accepts am HF dataset and tokenizer, tokenizes the text amd realigns the labels to the wordpieces.
    Returns the tokenized input which can be used with HF models
    The label for special tokens added by the 
    """
    tokenized_input = tokenizer(dataset["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    label_all_tokens = True
    for i, label in enumerate(dataset["tags"]):
        word_ids = tokenized_input.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_input["labels"] = labels
    return tokenized_input

def preprocess_file(reader, filename, label_vocab_file, task_id):
    """
    Accepts a filename in the UF format, uses the UF reader and returns a Huggingface dataset with two columns the tokens and corresponding NER tags 
    """
    with open(label_vocab_file, "r", encoding="utf-8") as f:
        labels = f.readlines()
    label_vocab = {label.rstrip(): i for i, label in enumerate(labels)}
    pred_to_label = {v: k for k, v in label_vocab.items()}

    with open(filename, "r") as f:
        input_lines = f.readlines()

    all_tokens = []
    all_tags = []
    all_ids = []
    for line in input_lines:
        # print(line)
        try:
            doc = reader.read_json(json.loads(line))
        except:
            print(line)
            doc = reader.read_json(json.loads(line))
        # print(doc)
        try:
            tokens, tags = zip(*[(tok.text, tok.attributes['tag']) for tok in doc.tokens])
        except:
            continue
       
        # print(tags)
        tags = list(map(label_vocab.get, tags))
        # print(list(set(tags)))
        assert None not in tags, print(tokens, tags)
        # print(tags)
        all_tokens.append(tokens)
        all_tags.append(tags)
        all_ids.append(task_id)
        # print(tokens, len(tokens))
        # print(tags, len(tags))
        
    dataset = {'tokens': all_tokens, 'tags':all_tags, 'task_ids':all_ids}

    dataset = Dataset.from_dict(dataset)

    print("Dataset from file: ", filename, ", Number of examples: ", len(dataset), ", Number of lines: ", len(input_lines))
    return dataset

def get_parser():
    parser = HfArgumentParser(()) # argparse.ArgumentParser("Basic single model fine tuning")
    parser.add_argument("--save-model", type=str, default="test-finetune", help="Location to save HF model")
    parser.add_argument("--model-type", type=str, default="xlm-roberta-base", help="Kind of model being finetuned")
    parser.add_argument("--target-task-args", required=True, type=str, help="Location to JSON file with saved args")
    parser.add_argument("--prefinetuning-args", required=False, type=str, help="Location to JSON file with prefinetuning args")
    parser.add_argument("--batch-size", default=8, type=int, help="Batch size for training")
    parser.add_argument("--num-epochs", default=5, type=int, help="Epochs to which the model should be trained")
    parser.add_argument("--use-encoder-weights", default=None, type=str, help="Shared weights to be used")
    parser.add_argument("--use-prefinetune-taskemb", default=None, type=str, help="Learned embedding to use")
    parser.add_argument("--encoding", default=None, type=str, help="Encoding used in the data - iob etc.")
    parser.add_argument("--local_rank", default=None, type=int, help="Encoding used in the data - iob etc.")
    return parser

def main():
    """
    Main driver function
    TODOs - further parameterize some of the input
    """
    parser = get_parser()
    args = parser.parse_args()

    if args.encoding is not None:
        reader = SeqTagDataReader((SeqTagTask.NER, ), encoding=args.encoding)
    else:
        reader = SeqTagDataReader((SeqTagTask.NER,))

    task = "ner"
    model_checkpoint = args.model_type # "xlm-roberta-base"
    batch_size = args.batch_size
    n_epochs = int(args.num_epochs)
    data_args = json.load(open(args.target_task_args, "r"))
    assert len(data_args["tasks"])==1, print("Use this script for target task finetuning not multiple tasks", data_args)
    # Borrowed from the Bert script
    with open(data_args["data_dir"]+data_args["task_label_vocab"][data_args["tasks"][0]], "r", encoding="utf-8") as f:
        labels = f.readlines()
    label_vocab = {label.rstrip(): i for i, label in enumerate(labels)}
    pred_to_label = {v: k for k, v in label_vocab.items()}

    dataset_list = []
    for i, task in enumerate(data_args["tasks"]):
        if data_args["task_kind"][i] == "seq":
            dataset_list.append(preprocess_file(reader, data_args["data_dir"]+data_args["task_train_data_map"][task], data_args["data_dir"]+data_args["task_label_vocab"][task], i))

    target_dataset = DatasetDict()
    target_dataset["train"] = concatenate_datasets(dataset_list).shuffle(seed=0)

    val_dataset_list = []
    for i, task in enumerate(data_args["tasks"]):
        if data_args["task_kind"][i] == "seq":
            val_dataset_list.append(preprocess_file(reader, data_args["data_dir"]+data_args["task_val_data_map"][task], data_args["data_dir"]+data_args["task_label_vocab"][task], i))

    target_dataset["validation"] = concatenate_datasets(val_dataset_list).shuffle(seed=0)

    print("Finished converting the datasets from UF to HF Datasets")
   
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenized_datasets = target_dataset.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer":tokenizer})
    print("Finished tokenizing text and aligning labels to word pieces")
    
    config = AutoConfig.from_pretrained(model_checkpoint)
    model = MultiheadModel.from_pretrained(model_checkpoint, mtl_args=data_args, config=config)
    if args.use_encoder_weights is not None:
        prefinetune_args = json.load(open(args.prefinetuning_args, "r"))
        mtl_encoder = MultiheadModel.from_pretrained(args.use_encoder_weights, mtl_args = prefinetune_args, config = config)
        setattr(model, "roberta", mtl_encoder.roberta)
        print("Loaded prefinetuned weights")

    train_args = TrainingArguments(
        args.save_model,
        save_strategy='epoch',
        evaluation_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=n_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [pred_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [pred_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "eval_f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    if args.save_model is not None:
        print("Model to be saved at: ", args.save_model)

    trainer = Trainer(
        model,
        train_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience = 3)]
    )
    trainer.train()
    if args.save_model is not None:
        model.save_pretrained(args.save_model)

if __name__=='__main__':
    main()
