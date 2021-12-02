#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for question answering.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.
import pdb
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
from datasets import load_dataset, load_metric

import transformers
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
    Trainer
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from utils_qa import postprocess_qa_predictions
from mstar.models.multihead_model.model import MultiheadModel

logger = logging.getLogger(__name__)

def get_qa_dataset(taskname):
    if taskname == "squad_adversarial":
        raw_datasets = load_dataset(taskname, "AddSent",)
        question_column_name = "question"
        answer_column_name = "answers"
        answer_start_name = "answer_start"
        context_column_name = "context"
    
        raw_datasets = raw_datasets['validation'].train_test_split(test_size = 0.2, shuffle=False, seed = 0)
    elif taskname == "subjqa":
        question_column_name = "question"
        answer_column_name = "answers"
        answer_start_name = "answer_start"
        context_column_name = "context"
    
        list_datasets = []
        for subset in ['books', 'electronics', 'grocery', 'movies', 'restaurants', 'tripadvisor']:
            list_datasets.append(load_dataset(taskname, subset))

        raw_datasets = datasets.DatasetDict()
        raw_datasets['train'] = datasets.concatenate_datasets([d['train'] for d in list_datasets])
        raw_datasets['validation'] = datasets.concatenate_datasets([d['validation'] for d in list_datasets])
        raw_datasets['test'] = datasets.concatenate_datasets([d['test'] for d in list_datasets])
        raw_datasets = raw_datasets.filter(lambda example: len(example['answers']['text']) > 0)
    elif taskname == "xquad_r":
        raw_datasets = load_dataset(taskname, "en",)
        question_column_name = "question"
        answer_column_name = "answers"
        answer_start_name = "answer_start"
        context_column_name = "context"
    
        raw_datasets = raw_datasets['validation'].train_test_split(test_size = 0.2, shuffle=False, seed = 0)
    elif taskname == "qed":
        loaded_datasets = load_dataset(taskname)
        def reformat(example):
            example['formatted_answers'] = {}
            try:
                example['formatted_answers']['answer_start'] = [example['annotation']['answer'][0]['paragraph_reference']['start']]
                example['formatted_answers']['text'] = [example['annotation']['answer'][0]['paragraph_reference']['string']]
            except:
                example['formatted_answers']['answer_start'] = []
                example['formatted_answers']['text'] = []
            example['id'] = str(example['example_id'])
            return example

        raw_datasets = loaded_datasets.map(reformat)
        raw_datasets = raw_datasets.filter(lambda example: len(example['formatted_answers']['answer_start'])>0)
        raw_datasets = raw_datasets.rename_column('paragraph_text', 'context')
        question_column_name = "question"
        answer_column_name = "formatted_answers"
        answer_start_name = "answer_start"
        context_column_name = "context"
    
        raw_datasets = raw_datasets['validation'].train_test_split(test_size = 0.2, shuffle=False, seed = 0)

    return raw_datasets, question_column_name, answer_column_name, context_column_name, answer_start_name

def get_parser():
    # parser = argparse.ArgumentParser("Basic single model fine tuning")
    parser = HfArgumentParser(()) # argparse.ArgumentParser("Basic single model fine tuning")
    parser.add_argument("--save-model", type=str, default="test-qa", help="Location to save HF model")
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
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = get_parser()
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = "INFO"#training_args.get_process_log_level()
    logger.setLevel(log_level)#log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # download the dataset.
    data_args = json.load(open(args.target_task_args, "r"))
    assert len(data_args['tasks']) == 1 and data_args['task_kind'][0] == "qa"
    dataset_name = data_args["tasks"][0] # is not None:
    raw_datasets, question_column_name, answer_column_name, context_column_name, answer_start_name = get_qa_dataset(dataset_name)
    print(raw_datasets)
    # download model & vocab.
    model_name = args.model_type #"xlm-roberta-base"
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name,
        config=config,
    )
    if args.use_encoder_weights is not None:
        prefinetuning_args = json.load(open(args.prefinetuning_args, "r"))
        mtl_encoder = MultiheadModel.from_pretrained(args.use_encoder_weights, mtl_args = prefinetuning_args, config = config)
        setattr(model, "roberta", mtl_encoder.roberta)

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    column_names = raw_datasets['train'].column_names
    # pdb.set_trace()

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    max_seq_length = tokenizer.model_max_length

    # Training preprocessing
    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # pdb.set_trace()
            # print(answers)
            # If no answers are given, set the cls_index as answer.
            # if len(answers["answer_start"]) == 0:
            if len(answers[answer_start_name]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                # start_char = answers["answer_start"][0]
                start_char = answers[answer_start_name][0]
                end_char = answers[answer_start_name][0] + len(answers['text'][0])
                #end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        # print(offsets[token_start_index][0], start_char)
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    train_dataset = raw_datasets["train"]
    train_dataset = train_dataset.map(
                prepare_train_features,
                batched=True,
                remove_columns=column_names,
                desc="Running tokenizer on train dataset",
            )

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        # pdb.set_trace()
        return tokenized_examples

    eval_examples = raw_datasets["test"]
    # pdb.set_trace()
    eval_dataset = eval_examples.map(
                prepare_validation_features,
                batched=True,
                remove_columns = column_names,
                desc="Running tokenizer on validation dataset",
            )
    # Data collator
    # collator.
    data_collator = (default_data_collator)

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=False, #data_args.version_2_with_negative,
            n_best_size=20, #data_args.n_best_size,
            max_answer_length=30, #data_args.max_answer_length,
            null_score_diff_threshold=0, #data_args.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            log_level=log_level,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
        # pdb.set_trace()
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction):
        # pdb.set_trace()
        labels = [{'id':item['id'], 'answers':{'text':item['answers']['text'], 'answer_start':item['answers']['answer_start'] }} for i, item in enumerate(p.label_ids)]
        scores =  metric.compute(predictions=p.predictions, references=labels)
        return scores # metric.compute(predictions=p.predictions, references=p.label_ids)

    batch_size = args.batch_size
    n_epochs = args.num_epochs
    training_args = TrainingArguments(
        args.save_model,
        save_strategy='epoch',
        evaluation_strategy = "epoch",
        # eval_steps=300,
        # prediction_loss_only=True,
        learning_rate=3e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=n_epochs,
        weight_decay=0.001,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        )
    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
    #trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=eval_examples,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # Training
    train_result = trainer.train()#resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
