# Training Tokenizers

## Requirements
* sentencepiece
* datasets

## Usage
Use `compute_bytes.py` to compute byte-wise statistics of your training data. This will be useful to inform the value of `--max_sentence_length` below.

**Example command:**
```
python compute_bytes.py --data train.arrow
```


Use `train_unigram.py` to train a unigramLM tokenizer. The script assumes the dataset is in arrow format. `--max_sentence_length` uses bytes as units and will result in the trainer throwing away all examples longer than this value.

**Example command:**
```
python train_tokenizer.py \
--data /mnt/pile_mnt/pile_arrow_no_youtube_no_exact_match/train.arrow \
--byte_fallback \
--vocab_size 50000 \
-num_examples 10000000 \
--name_suffix pile_no_youtube_no_exact_match \
--max_sentence_length 16384 \
--output_folder output
```


Use `compute_compression_stats.py` to measure the compression ability of your tokenizer after training. Metrics are {mean, median, min, max} tokenized sequence length and tokens per byte. `--subset` extracts a subset of examples that longer than the value passed to the argument, in bytes.

**Example command:**
```
python compute_compression_stats.py --data val.arrow --tokenizer unigramlm_pile_no_youtube_no_exact_match_msl-16384_n-10000000_v-50000.model
```
