from .t5_collator import (
    compute_input_and_target_lengths,
    T5DataCollatorForSpanCorruption,
    BARTDataCollatorForSpanCorruption,
)
import transformers
import logging

logger = logging.getLogger(__name__)


def get_collator(data_args, decoder_start_token_id):

    expanded_inputs_length, target_length = compute_input_and_target_lengths(
        inputs_length=data_args.max_seq_length,
        noise_density=data_args.mlm_prob,
        mean_noise_span_length=data_args.mean_noise_span,
    )

    # if we run over the sentinel ids for span corruption we will use normal tokens as sentinel ids, which is a silent error
    sentinel_ids_needed = int(
        data_args.max_seq_length
        * (0.1 + data_args.mlm_prob)
        / data_args.mean_noise_span
    )

    if data_args.tokenizer == "t5-base":
        logger.info("Loading t5-base tokenizer")
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "t5-base",
            extra_ids=sentinel_ids_needed,
            model_max_length=expanded_inputs_length,
        )
    elif "unigramlm_easel_n" in data_args.tokenizer:
        logger.info(f"Loading custom tokenizer {data_args.tokenizer}")
        # custom mstar tokenizer
        import mstar.tokenizers.sentencepiece

        # tokenizer = mstar.tokenizers.sentencepiece.Sentencepiece(vocab_file='/mnt/tokenizer/unigramlm_easel_n-1000000_v-32000.model')
        tokenizer = mstar.tokenizers.sentencepiece.SentencepieceTokenizer(
            vocab_file=data_args.tokenizer,
            extra_ids=sentinel_ids_needed,
            model_max_length=expanded_inputs_length,
        )

    if data_args.collator_output_targets == "t5":
        collator = T5DataCollatorForSpanCorruption(
            tokenizer=tokenizer,
            noise_density=data_args.mlm_prob,
            mean_noise_span_length=data_args.mean_noise_span,
            expandend_inputs_length=expanded_inputs_length,
            input_length=data_args.max_seq_length,
            target_length=target_length,
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=decoder_start_token_id,
        )

    elif data_args.collator_output_targets == "bart":
        collator = BARTDataCollatorForSpanCorruption(
            tokenizer=tokenizer,
            noise_density=data_args.mlm_prob,
            mean_noise_span_length=data_args.mean_noise_span,
            expandend_inputs_length=expanded_inputs_length,
            input_length=data_args.max_seq_length,
            target_length=target_length,
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=decoder_start_token_id,
        )

    else:
        raise NotImplementedError()

    return tokenizer, collator
