import argparse
import datasets
import sentencepiece as spm
from operator import truediv
import pandas as pd

pd.set_option('display.max_columns', None)

parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', type=str)
parser.add_argument('--tokenizer', '-t', type=str)
parser.add_argument('--subset', '-s', type=int, default=0)

args = parser.parse_args()

data = datasets.arrow_dataset.Dataset.from_file(args.data)

tokenizer = spm.SentencePieceProcessor(model_file=args.tokenizer)

def tokenize(examples):
    examples['tokens'] = tokenizer.encode(examples['text'], out_type=int)
    return examples

def compute_tokens(examples):
    return {'num_tokens': [len(t) for t in examples['tokens']]}

def compute_bytes(examples):
    examples['num_bytes'] = [len(s.encode('utf-8')) for s in examples['text']]
    return examples

def compute_tokens_per_byte(examples):    
    return {'tokens_per_byte': list(map(truediv, examples['num_tokens'], examples['num_bytes']))}

def print_stats(df, key):
    print(key)
    print('Mean:', f'{df[key].mean():.4f}')
    print('St Dev:', f'{df[key].std():.4f}')
    print('Median:', f'{df[key].median():.4f}')
    print('Min:', f'{df[key].min():.4f}')
    print('Max:', f'{df[key].max():.4f}')
    print()

data = data.map(compute_bytes, batched=True, batch_size=1000, num_proc=50, cache_file_name="/tmp/cache.arrow")
print(data.column_names)

if args.subset > 0:
    data = data.filter(lambda example: example['num_bytes'] > args.subset, num_proc=50)


df = data\
    .map(tokenize, batched=True, batch_size=1000, num_proc=50)\
    .map(compute_tokens, batched=True, batch_size=1000, num_proc=50)\
    .map(compute_tokens_per_byte, batched=True, batch_size=1000, num_proc=50)\
    .to_pandas()

print(df[df['tokens_per_byte'] > 1][['text', 'tokens', 'tokens_per_byte']])
print(df[df['num_tokens'] == 1][['text', 'tokens', 'tokens_per_byte']])

print_stats(df, 'num_tokens')
print_stats(df, 'num_bytes')
print_stats(df, 'tokens_per_byte')
