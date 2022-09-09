import argparse
import datasets

parser = argparse.ArgumentParser()
parser.add_argument('--data', '-d', type=str)

args = parser.parse_args()

data = datasets.arrow_dataset.Dataset.from_file(args.data, in_memory=True)


df = data.map(lambda examples: {'bytes': [len(s.encode('utf-8')) for s in examples['text']]}, batched=True, batch_size=5000, num_proc=10, remove_columns=['text']).to_pandas()

print('Mean:', df['bytes'].mean())
print('Median:', df['bytes'].median())
print('St Dev:', f'{df["bytes"].std():.4f}')
print('Max:', df['bytes'].max())
print('Min:', df['bytes'].min())

