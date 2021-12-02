import numpy as np
from transformers import HfArgumentParser
import json
import torch
import re
import argparse
from mtl_trainer import MultitaskTrainer
from mstar.uf_format.uf_reader import SeqTagTask, SeqTagDataReader
from transformers import AutoTokenizer
from transformers import EarlyStoppingCallback
from transformers import AutoModelForTokenClassification, AutoConfig, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from datasets import load_metric, Dataset, DatasetDict, concatenate_datasets, interleave_datasets
from mstar.models.task_embedding_model.model import TaskEmbeddingMultiheadModel
from mstar.models.multihead_model.model import MultiheadModel
from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset
import pdb
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
    tokenized_input["labels"] = labels
    tokenized_input["glue_labels"] = dataset["glue_labels"]

    tokenized_input["start_positions"] = []
    tokenized_input["end_positions"] = []
    c = 0
    for i, span_start in enumerate(dataset['qa_spans_start']):
        input_ids = tokenized_input['input_ids'][i]

        ans_start = dataset["qa_spans_start"][i]
        ans_end = dataset["qa_spans_end"][i]
        if ans_start == -1 or ans_end == -1:
            tokenized_input["start_positions"].append([-1])
            tokenized_input["end_positions"].append([-1])
            continue

        start_idx = 0
        end_idx = 0
        word_ids = tokenized_input.word_ids(batch_index=i)
        for idx in range(len(word_ids)-1):
            if start_idx == 0 and word_ids[idx] == ans_start:
                start_idx = idx
            if end_idx == 0 and word_ids[idx] == ans_end and word_ids[idx+1] != ans_end:
                end_idx = idx
        
        tokenized_input["start_positions"].append([start_idx])
        tokenized_input["end_positions"].append([end_idx+1])
        if start_idx == 0 and end_idx == 0:
            c+=1

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
        try:
            tokens, tags = zip(*[(tok.text, tok.attributes['tag']) for tok in doc.tokens])
        except:
            continue
       
        tags = list(map(label_vocab.get, tags))
        assert None not in tags, print(tokens, tags)
        all_tokens.append(tokens)
        all_tags.append(tags)
        all_ids.append(task_id)
        
    all_labels = [-1 for i in range(len(all_ids))]
    all_spans_start = [-1 for i in range(len(all_ids))]
    all_spans_end = [-1 for i in range(len(all_ids))]
    dataset = Dataset.from_dict({'tokens': all_tokens, 'tags':all_tags, 'task_ids':all_ids, 'glue_labels':all_labels, 'qa_spans_start':all_spans_start, 'qa_spans_end':all_spans_end})#, 'qa_spans':all_spans})

    print("Dataset from file: ", filename, ", Number of examples: ", len(dataset), ", Number of lines: ", len(input_lines))
    print(dataset.features.type)
    return dataset

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

    all_ids = [task_id for i in range(len(all_tokens))]
        
    all_tags = [[-1] for i in range(len(all_ids))]
    all_spans_start = [-1 for i in range(len(all_ids))]
    all_spans_end = [-1 for i in range(len(all_ids))]
    dataset = Dataset.from_dict({'tokens': all_tokens, 'tags':all_tags, 'task_ids':all_ids, 'glue_labels':all_labels, 'qa_spans_start':all_spans_start, 'qa_spans_end':all_spans_end})#, 'qa_spans':all_spans})
    print(dataset.features.type)

    print("Dataset from task: ", taskname, ", Number of examples: ", len(dataset))
    return dataset


def get_qa_dataset(task, split, task_id):
    if task != "mrqa":
        return 
    all_tokens = []
    all_tags = []
    all_labels = []
    all_spans_start = []
    all_spans_end = []
    all_spans = []
    dataset = load_dataset(task, split = split)
    count_failed = 0
    for i in range(len(dataset)):
        if i%50000 == 0:
            print(i)
        text = dataset[i]['question'] + " </s> </s> " + dataset[i]['context']
        # tokens = re.split(',. )(', text) #text.split()
        text = text.replace(",", " , ")
        text = text.replace(".", " . ")
        text = text.replace(")", " ) ")
        text = text.replace("(", " ( ")
        text = text.replace('"', ' " ')
        text = text.replace("'", " ' ")
        tokens = text.split()
        ans_text = dataset[i]['detected_answers']['text'][0]
        ans_tokens = ans_text.split()
        def find_sub_list(sl,l):
            sll=len(sl)
            for ind in (i for i,e in enumerate(l) if e==sl[0]):
                if l[ind:ind+sll]==sl:
                    return ind,ind+sll-1
        start, end = -1, -1
        try:
            start, end = find_sub_list(ans_tokens, tokens)
        except:
            try:
                start, end = find_sub_list([ans_tokens[0]], tokens)
            except:
                count_failed += 1
        if start!= -1:
            all_tokens.append(tokens)
            all_spans_start.append(start)
            all_spans_end.append(end)
            all_spans.append(ans_text)

    all_ids = [task_id for i in range(len(all_tokens))]
    all_labels = [-1 for i in range(len(all_ids))]
    all_tags = [[-1] for i in range(len(all_ids))]
    dataset = Dataset.from_dict({'tokens': all_tokens, 'tags':all_tags, 'task_ids':all_ids, 'glue_labels':all_labels, 'qa_spans_start':all_spans_start, 'qa_spans_end':all_spans_end})#, 'qa_spans':all_spans})
    print(dataset.features.type)

    print("Dataset from task: ", task, ", Number of examples: ", len(dataset), " and failed ", count_failed)
    return dataset

def get_parser():
    parser = HfArgumentParser(()) # argparse.ArgumentParser("Basic single model fine tuning")
    parser.add_argument("--save-model", type=str, default="test-multihead", help="Location to save HF model")
    parser.add_argument("--data-args", type=str, required=True, help="Arguments file")
    parser.add_argument("--model-type", type=str, default="xlm-roberta-base", help="Model type on HF")
    parser.add_argument("--batch-size", default=8, type=int, help="Batch size for training")
    parser.add_argument("--num-epochs", default=5, type=int, help="Epochs to which the model should be trained")
    parser.add_argument("--use-encoder-weights", default=None, type=str, help="Shared weights to be used")
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

    data_args = json.load(open(args.data_args, "r"))

    model_checkpoint = args.model_type#"xlm-roberta-base"
    batch_size = args.batch_size
    n_epochs = int(args.num_epochs)

    dataset_list = []
    prefinetune_dataset = DatasetDict()

    for i, task in enumerate(data_args["tasks"]):
        if data_args["task_kind"][i] == "seq":
            dataset_list.append(preprocess_file(reader, data_args["data_dir"]+data_args["task_train_data_map"][task], data_args["data_dir"]+data_args["task_label_vocab"][task], i))
        elif data_args["task_kind"][i] == "glue":
            dataset_list.append(get_glue_dataset(task, "train", i))
        elif data_args["task_kind"][i] == "qa":
            dataset_list.append(get_qa_dataset(task, "train", i))

    prefinetune_dataset["train"] = concatenate_datasets(dataset_list).shuffle(seed=0)
    val_dataset_list = []
    for i, task in enumerate(data_args["tasks"]):
        if data_args["task_kind"][i] == "seq":
            val_dataset_list.append(preprocess_file(reader, data_args["data_dir"]+data_args["task_val_data_map"][task], data_args["data_dir"]+data_args["task_label_vocab"][task], i))
        elif data_args["task_kind"][i] == "glue":
            val_dataset_list.append(get_glue_dataset(task, "validation", i))
        elif data_args["task_kind"][i] == "qa":
            val_dataset_list.append(get_qa_dataset(task, "validation", i))

    prefinetune_dataset["validation"] = concatenate_datasets(val_dataset_list).shuffle(seed=0)
    print("Finished converting the datasets from UF to HF Datasets")
   
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    tokenized_datasets = prefinetune_dataset.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer":tokenizer})
    print("Finished tokenizing text and aligning labels to word pieces")

    config = AutoConfig.from_pretrained(model_checkpoint)
    model = MultiheadModel.from_pretrained(model_checkpoint, mtl_args=data_args, config=config)

    train_args = TrainingArguments(
        args.save_model,
        save_strategy='epoch',
        evaluation_strategy = "epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=4,
        num_train_epochs=n_epochs,
        weight_decay=0.001,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        )   

    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        true_predictions = [
            [str(p.item()) for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [str(l.item()) for (p, l) in zip(prediction, label) if l != -100]
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
    trainer = MultitaskTrainer(
    # trainer = Trainer(
        model,
        train_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience = 3)],
    )
    trainer.train()
    if args.save_model is not None:
        model.save_pretrained(args.save_model)

if __name__=='__main__':
    main()
