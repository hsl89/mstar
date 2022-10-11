from torch.utils.data import Dataset, DataLoader
import os
import math
import copy
import torch
import transformers
from transformers import T5Tokenizer
from dataclasses import dataclass, field
from transformers import (
    T5ForConditionalGeneration,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    AutoConfig,
    T5Config,
)
from transformers import TrainingArguments, HfArgumentParser

# from transformers.models.longformer import LongformerSelfAttention
import argparse
import pyarrow as pa
import warnings

# from transformers.models.bert.modeling_bert import BertSelfAttention
from transformers import T5Tokenizer
import torch.nn as nn

warnings.simplefilter("ignore")

MAX_STEPS = 5
MAX_SEQ_LENGTH = 2048
MLM_PROB = 0.15
MEAN_NOISE_SPAN = 3.0
EVAL_ONLY = False
torch.manual_seed(1)


class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_indices):
        self.dataset_indices = dataset_indices

    def __getitem__(self, index):
        # TODO: just use dhananjay's
        pa_index = pa.array(index if type(index) == list else [index])
        return self.dataset_indices.take(pa_index)["text"].to_pylist()

    def __len__(self):
        return len(self.dataset_indices)


INDICES = [11, 10101, 99999, 12]
TRAIN_DATA_PATH = "/mnt/colehawk/pile_no_youtube/val_packed_chunksize_2600.arrow"  # where are train/test/val datasets
mmap = pa.memory_map(TRAIN_DATA_PATH)
train_dataset = pa.ipc.open_stream(mmap).read_all()

# Check table has single chunk
# https://issues.apache.org/jira/browse/ARROW-11989
assert len(train_dataset["text"].chunks) == 1

batch = train_dataset.take(INDICES)["text"].to_pylist()


import collators

"""
tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token':'[PAD]'})
data_collator = collators.gpt2_data_collator.DataCollatorForGPT(
    tokenizer, padding=True, max_length=2048,pad_to_multiple_of=None)
"""


model = transformers.T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "t5-base", extra_ids=500, model_max_length=(1 + 0.05 + MLM_PROB) * MAX_SEQ_LENGTH
)
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
model.resize_token_embeddings(len(tokenizer))
config = transformers.AutoConfig.from_pretrained("t5-base")

(
    expanded_inputs_length,
    target_length,
) = collators.t5_collator.compute_input_and_target_lengths(
    inputs_length=2048, noise_density=MLM_PROB, mean_noise_span_length=MEAN_NOISE_SPAN
)
print("Inputs lengths", expanded_inputs_length, target_length)
"""
data_collator = collators.t5_collator.T5DataCollatorForSpanCorruption(
            tokenizer=tokenizer,
            noise_density=MLM_PROB,
            mean_noise_span_length=MEAN_NOISE_SPAN,
            expandend_inputs_length=expanded_inputs_length,
            input_length=2048,
            target_length=target_length,
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=config.decoder_start_token_id)
"""
data_collator = collators.t5_collator.BARTDataCollatorForSpanCorruption(
    tokenizer=tokenizer,
    noise_density=MLM_PROB,
    mean_noise_span_length=MEAN_NOISE_SPAN,
    expandend_inputs_length=expanded_inputs_length,
    input_length=2048,
    target_length=target_length,
    pad_token_id=tokenizer.pad_token_id,
    decoder_start_token_id=config.decoder_start_token_id,
)

for i, x in enumerate(batch):
    print(i, type(x), x[:10])

out = data_collator(batch)
print(out.keys())
print("Encoder Input ids \n", out["input_ids"][:, :10])
print("last Encoder Input ids \n", out["input_ids"][:, -10:])
# print("last Encoder attn mask",out['attention_mask'][:,-15:])
print("Decoder Last Input ids \n", out["decoder_input_ids"][:, -10:])
print("First Labels \n", out["labels"][:, :10])
print("First dec ma \n", out["decoder_attention_mask"][:, :10])
print("kast dec  ma \n", out["decoder_attention_mask"][:, -100:])
print("Last Labels \n", out["labels"][:, -100:])

model = model.cuda()
out = out.to("cuda")
with torch.no_grad():
    loss = model(
        input_ids=out["input_ids"],
        labels=out["labels"],
        decoder_input_ids=out["decoder_input_ids"],
    ).loss

print(loss)

# loss.backward()

"""
def pretrain_and_evaluate(args, model, tokenizer, eval_only, model_path):
    training_dataset_path = 'pile/training.arrow'
    expandend_inputs_length, target_length = data_processing.compute_input_and_target_lengths(
        inputs_length=MAX_SEQ_LENGTH,
        noise_density=MLM_PROB,
        mean_noise_span_length=MEAN_NOISE_SPAN_LEN,
    )
    setattr(tokenizer,'model_max_length',expandend_inputs_length)

    data_collator = data_processing.T5DataCollatorForMLM(
                                            tokenizer=tokenizer,
                                            noise_density=MLM_PROB,
                                            mean_noise_span_length=MEAN_NOISE_SPAN_LEN,
                                            expandend_inputs_length=expandend_inputs_length,
                                            input_length=MAX_SEQ_LENGTH,#79*128, #blocksize multiple
                                            target_length=target_length,
                                            pad_token_id=model.config.pad_token_id,
                                            decoder_start_token_id=model.config.decoder_start_token_id,
                                        )
    train_index_dataset = train_dataset

    #DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)#, mlm_probability=0.15)
    trainer = Trainer(model=model, args=args, train_dataset=train_index_dataset,eval_dataset = train_index_dataset,data_collator=data_collator)

    eval_loss = trainer.evaluate()
    eval_loss = eval_loss['eval_loss']
    print('Initial eval bpc: {}'.format(eval_loss/math.log(2)))
    
    if not eval_only:
        trainer.train(model_path=model_path)
        trainer.save_model()

        eval_loss = trainer.evaluate()
        eval_loss = eval_loss['eval_loss']
        print('Eval bpc after pretraining: {}'.format(eval_loss/math.log(2)))


@dataclass
class ModelArgs:
    max_pos: int = field(default=MAX_SEQ_LENGTH, metadata={"help": "Maximum position"})

parser = HfArgumentParser((TrainingArguments, ModelArgs,))


training_args, model_args = parser.parse_args_into_dataclasses(look_for_args_file=False, args=[
    '--output_dir', 'tmp',
    '--warmup_steps', '500',
    '--learning_rate', '0.00003',
    '--weight_decay', '0.01',
    '--adam_epsilon', '1e-6',
    #'--tf32',
    '--fp16',
    '--max_steps', '100',
    '--logging_steps', '500',
    '--save_steps', '500',
    '--max_grad_norm', '1.0',
    '--per_gpu_eval_batch_size', '2',
    '--per_gpu_train_batch_size', '2',  # 32GB gpu with fp32
    '--gradient_accumulation_steps', '6',
    '--do_train',
    '--do_eval',
])
training_args.val_datapath = '.datasets/wikitext-103-raw/wiki.valid.raw'
training_args.train_datapath = '.datasets/wikitext-103-raw/wiki.train.raw'
training_args.train_datapath = training_args.val_datapath #'.datasets/wikitext-103-raw/wiki.valid.raw'

model_path = f'{training_args.output_dir}/roberta-base-{model_args.max_pos}'
if not os.path.exists(model_path):
    os.makedirs(model_path)

#print(model_args.max_pos)
model_name = 't5-small'

config = AutoConfig.from_pretrained(model_name)
print(config)

config = transformers.AutoConfig.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)
from argparse import Namespace
sparsity_config = Namespace(**{'block_size':BLOCKSIZE,'max_seq_len':MAX_SEQ_LENGTH}) 
model = models.T5LongForConditionalGeneration(config,sparsity_config=sparsity_config)#.from_pretrained(model_name)
#model = T5LongForConditionalGeneration(config,sparsity_config=sparsity_config)#.from_pretrained(model_name)
pretrain_and_evaluate(training_args, model, tokenizer, eval_only=EVAL_ONLY, model_path=None)	

"""
