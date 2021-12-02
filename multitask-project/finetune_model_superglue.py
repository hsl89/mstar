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
from transformers import DataCollatorForTokenClassification, DataCollatorWithPadding
from datasets import load_metric, load_dataset, Dataset, DatasetDict, concatenate_datasets
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
    # tokenized_input = tokenizer(dataset["tokens"], truncation=True, padding='max_length', max_length=128, is_split_into_words=True)
    labels = []
    label_all_tokens = True
    glue_labels = []
    for i, label in enumerate(dataset["tags"]):
        if len(label) > 1:
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
        else:
            labels.append(label)
    tokenized_input["labels"] = dataset["glue_labels"]
    return tokenized_input

def get_glue_dataset(taskname, split, task_id):
    all_tokens = []
    all_labels = []
    if taskname == "cola":
        dataset = load_dataset("glue", taskname, split=split)
        for i in range(len(dataset)):
            all_tokens.append(dataset[i]['sentence'].split())
            all_labels.append(dataset[i]['label'])
    elif taskname == "mnli_matched" or taskname == "mnli_mismatched":
        if split == "train":
            dataset = load_dataset("glue", "mnli", split=split)
            if taskname == "mnli_matched":
                dataset = dataset.filter(lambda example, index: index % 2 == 0, with_indices=True)
            else:
                dataset = dataset.filter(lambda example, index: index % 2 == 1, with_indices=True)
        else:
            dataset = load_dataset("glue", taskname, split=split)

        for i in range(len(dataset)):
            text = dataset[i]['premise'] + " </s> </s> " + dataset[i]['hypothesis']
            all_tokens.append(text.split())
            all_labels.append(dataset[i]['label'])
    elif taskname == "qnli":
        dataset = load_dataset("glue", taskname, split = split)
        for i in range(len(dataset)):
            text = dataset[i]['question'] + " </s> </s> " + dataset[i]['sentence']
            all_tokens.append(text.split())
            all_labels.append(dataset[i]['label'])
    elif taskname == "wnli" or taskname == "mrpc" or taskname == "rte" or taskname == "stsb":
        dataset = load_dataset("glue", taskname, split = split)
        for i in range(len(dataset)):
            text = dataset[i]['sentence1'] + " </s> </s> " + dataset[i]['sentence2']
            all_tokens.append(text.split())
            all_labels.append(dataset[i]['label'])
    elif taskname == "sst2":
        dataset = load_dataset("glue", taskname, split=split)
        for i in range(len(dataset)):
            all_tokens.append(dataset[i]['sentence'].split())
            all_labels.append(dataset[i]['label'])
    elif taskname == "qqp":
        dataset = load_dataset("glue", taskname, split = split)
        for i in range(len(dataset)):
            text = dataset[i]['question1'] + " </s> </s> " + dataset[i]['question2']
            all_tokens.append(text.split())
            all_labels.append(dataset[i]['label'])
    elif taskname == "cb":
        dataset = load_dataset("super_glue", taskname, split=split)
        for i in range(len(dataset)):
            text = dataset[i]['premise'] + " </s> </s> " + dataset[i]['hypothesis']
            all_tokens.append(text.split())
            all_labels.append(dataset[i]['label'])
    elif taskname == "boolq":
        dataset = load_dataset("super_glue", taskname, split=split)
        for i in range(len(dataset)):
            text = dataset[i]['question'] + " </s> </s> " + dataset[i]['passage']
            all_tokens.append(text.split())
            all_labels.append(dataset[i]['label'])
    elif taskname == "copa":
        dataset = load_dataset("super_glue", taskname, split=split)
        for i in range(len(dataset)):
            text = dataset[i]['premise'] + " </s> </s> " + dataset[i]['choice1'] +  " </s> </s> " + dataset[i]['choice2']
            all_tokens.append(text.split())
            all_labels.append(dataset[i]['label'])
    elif taskname == "wic":
        dataset = load_dataset("super_glue", taskname, split=split)
        for i in range(len(dataset)):
            text = dataset[i]['sentence1'] + " </s> </s> " + dataset[i]['sentence2'] + " </s> </s> " + dataset[i]['word']
            all_tokens.append(text.split())
            all_labels.append(dataset[i]['label'])

    all_ids = [task_id for i in range(len(all_tokens))]
        
    # print(all_tokens[0], all_tags[0], all_ids[0])
    all_tags = [[-1] for i in range(len(all_ids))]
    dataset = Dataset.from_dict({'tokens': all_tokens, 'tags':all_tags, 'task_ids':all_ids, 'glue_labels':all_labels})
    print(dataset.features.type)

    print("Dataset from task: ", taskname, ", Number of examples: ", len(dataset))
    # print(dataset)
    # print(dataset[:3])
    # tokenize_and_align_labels(dataset, tokenizer)
    return dataset

def get_parser():
    parser = HfArgumentParser(()) # argparse.ArgumentParser("Basic single model fine tuning")
    parser.add_argument("--save-model", type=str, default="test-finetune-superglue", help="Location to save HF model")
    parser.add_argument("--model-type", type=str, default="xlm-roberta-base", help="Kind of model being finetuned")
    parser.add_argument("--target-task-args", required=True, type=str, help="Location to JSON file with saved args")
    parser.add_argument("--prefinetuning-args", type=str, help="Location to JSON file with prefinetuning args")
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

    model_checkpoint = args.model_type #"xlm-roberta-base"
    batch_size = args.batch_size
    n_epochs = int(args.num_epochs)
    data_args = json.load(open(args.target_task_args, "r"))
    assert len(data_args["tasks"])==1, print("Use this script for target task finetuning not multiple tasks", data_args)
    task_name = data_args['tasks'][0]
    print(task_name)

    dataset_list = []
    for i, task in enumerate(data_args["tasks"]):
        if data_args["task_kind"][i] == "glue":
            dataset_list.append(get_glue_dataset(task, "train", i))

    target_dataset = DatasetDict()
    target_dataset["train"] = concatenate_datasets(dataset_list).shuffle(seed=0)

    val_dataset_list = []
    for i, task in enumerate(data_args["tasks"]):
        if data_args["task_kind"][i] == "glue":
            val_dataset_list.append(get_glue_dataset(task, "validation", i))

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
        metric_for_best_model="accuracy" if task_name!="cb" else "f1",
        )

    data_collator = DataCollatorWithPadding(tokenizer)
    metric = load_metric("super_glue", data_args['tasks'][0])

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        if len(set(preds)) == 1:
            print("Always predicting one class ", set(preds))
        results = metric.compute(predictions = preds, references = p.label_ids)
        return results #{"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

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
