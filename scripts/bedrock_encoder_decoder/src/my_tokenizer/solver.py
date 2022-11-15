"""
Wrapper class to set up tokenizers
"""
import transformers
import logging
import mstar.tokenizers.sentencepiece

logger = logging.getLogger(__name__)


def get_tokenizer(data_args):

    # add extra tokens for easier downstream use
    sentinel_ids_needed = data_args.extra_tokenizer_ids

    # all span corruption tasks require masking tokens
    # if we run over the sentinel ids for span corruption
    # we will use normal tokens as sentinel ids
    # which is a silent error
    if data_args.mlm_prob is not None:
        sentinel_ids_needed += int(
            data_args.max_seq_length * data_args.mlm_prob / data_args.mean_noise_span
        )

    if data_args.tokenizer == "t5-base":
        logger.info("Loading t5 tokenizer")
        # rely on collator for truncation
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "t5-base",
            extra_ids=sentinel_ids_needed,
            model_max_length=2 * data_args.max_seq_length,
        )
        logger.info(f"Adding {sentinel_ids_needed} extra ids to the tokenizer")

    elif "unigramlm_easel_n" in data_args.tokenizer:
        raise ValueError("Old tokenizer")
        logger.info(f"Loading custom tokenizer from {data_args.tokenizer}")
        # custom mstar tokenizer
        # tokenizer = mstar.tokenizers.sentencepiece.Sentencepiece(vocab_file='/mnt/tokenizer/unigramlm_easel_n-1000000_v-32000.model')
        tokenizer = mstar.tokenizers.sentencepiece.SentencepieceTokenizer(
            vocab_file=data_args.tokenizer,
            extra_ids=sentinel_ids_needed,
            model_max_length=2 * data_args.max_seq_length,
        )
        logger.info(f"Adding {sentinel_ids_needed} extra ids to the tokenizer")

    elif data_args.tokenizer == "t5_subword_sampling":
        logging.info("Using t5 tokenizer with subword sampling")
        tokenizer = mstar.tokenizers.sentencepiece.SentencepieceTokenizer(
            vocab_file="/mnt/tokenizer/t5-base-sentencepiece.model",
            extra_ids=sentinel_ids_needed,
            model_max_length=1.2 * data_args.max_seq_length,
            sample_subwords=True,
            sampling_size=4,
        )

    elif data_args.tokenizer == "pile_50k_deduped_subword":
        tokenizer = mstar.tokenizers.sentencepiece.SentencepieceTokenizer(
            vocab_file="/mnt/tokenizer/unigramlm_pile_no_youtube_no_exact_match_msl-59999_n-10000000_v-50000.model",
            extra_ids=sentinel_ids_needed,
            model_max_length=2 * data_args.max_seq_length,
            sample_subwords=True,
        )

    else:
        raise ValueError

    return tokenizer
