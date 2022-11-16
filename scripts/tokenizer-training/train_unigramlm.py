import datasets
import sentencepiece as spm
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--vocab_size", "-v", type=int, required=True)
parser.add_argument("--num_examples", "-n", type=int, required=False, default=10000000)
parser.add_argument("--byte_fallback", "-b", action="store_true")
parser.add_argument("--max_sentence_length", "-msl", type=int, required=False, default=4096)
parser.add_argument("--data", "-d", type=str, nargs="+", required=True)
parser.add_argument("--name_suffix", "-s", type=str, required=False, default='')
parser.add_argument("--output_folder", "-o", type=str, required=False, default='output')
parser.add_argument("--num_threads", "-th", type=int, required=False, default=128)


args = parser.parse_args()

data = datasets.concatenate_datasets([datasets.arrow_dataset.Dataset.from_file(d) for d in args.data]).shuffle(seed=42).select(range(args.num_examples*2))

def dataset_iterator(dataset):
    for i in range(len(dataset)):
        yield dataset[i]["text"]

def make_folder(folder_path):
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

def format_model_path(directory: str, model: str, name: str, max_sentence_length: int, num_examples: int, vocab_size: int) -> str:
        return os.path.join(directory, '_'.join([model, name, 'msl-' + str(max_sentence_length), 'n-' + str(num_examples), 'v-' + str(vocab_size)]))

make_folder(args.output_folder)

save_location = format_model_path(args.output_folder, 'unigramlm', args.name_suffix, args.max_sentence_length, args.num_examples, args.vocab_size)

print("training started, saving at:", save_location)

spm.SentencePieceTrainer.train(
    sentence_iterator=dataset_iterator(data),
    model_prefix=save_location,
    vocab_size=args.vocab_size,
    model_type='unigram',
    input_sentence_size=args.num_examples,
    max_sentence_length=args.max_sentence_length,#999999
    num_threads=args.num_threads,
    byte_fallback=args.byte_fallback,
    train_extremely_large_corpus=True,
    pad_id=3,
    unk_surface="<unk>",
)
