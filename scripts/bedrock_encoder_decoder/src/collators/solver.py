from .t5_collator import (
    compute_input_and_target_lengths,
    T5DataCollatorForSpanCorruption,
    BARTDataCollatorForSpanCorruption,
    Seq2SeqMixDenoisingCollator,
    CLMCollator,
    MixedT5DataCollatorForSpanCorruption,
)
import transformers
import logging

logger = logging.getLogger(__name__)


def get_collator(data_args, tokenizer, decoder_start_token_id):

    if data_args.source == "mtl":
        # don't need collator for existing MLT pipeline
        return None

    if data_args.collator_output_targets == "t5":
        expanded_inputs_length, target_length = compute_input_and_target_lengths(
            inputs_length=data_args.max_seq_length,
            noise_density=data_args.mlm_prob,
            mean_noise_span_length=data_args.mean_noise_span,
        )
        collator = T5DataCollatorForSpanCorruption(
            tokenizer=tokenizer,
            noise_density=data_args.mlm_prob,
            mean_noise_span_length=data_args.mean_noise_span,
            expandend_inputs_length=expanded_inputs_length,
            input_length=data_args.max_seq_length,
            target_length=target_length,
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=decoder_start_token_id,
            clm_token="<extra_id_0>", #Passing CLM token to check that it is not being used as sentinel ID
        )

    elif data_args.collator_output_targets == "bart":

        collator = BARTDataCollatorForSpanCorruption(
            tokenizer=tokenizer,
            noise_density=data_args.mlm_prob,
            mean_noise_span_length=data_args.mean_noise_span,
            expandend_inputs_length=data_args.max_seq_length,
            input_length=data_args.max_seq_length,
            target_length=-1,  # arg has no effect
            keep_sentinel_ids=data_args.keep_sentinel_ids,
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=decoder_start_token_id,
        )

    elif data_args.collator_output_targets == "mixed_bart_clm":
        collator = Seq2SeqMixDenoisingCollator(
            tokenizer=tokenizer,
            max_length=data_args.max_seq_length,  # expanded_inputs_length,
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=decoder_start_token_id,
            noise_density=data_args.mlm_prob,
            mean_noise_span_length=data_args.mean_noise_span,
            mix_ratio=data_args.clm_mix_ratio,
            clm_max_doc=data_args.clm_max_doc,
            keep_sentinel_ids=data_args.keep_sentinel_ids,
        )

    elif data_args.collator_output_targets == "clm":
        collator = CLMCollator(
            tokenizer=tokenizer,
            max_input_length=data_args.max_seq_length,
            max_output_length=data_args.max_output_length,
            decoder_start_token_id=decoder_start_token_id,
            clm_token="<extra_id_0>",
            clm_min_split_ratio=0.2,  # same as AlexaTM
            clm_max_split_ratio=0.8,  # same as AlexaTM
        )

    elif data_args.collator_output_targets == "mixed_t5_clm":
        expanded_inputs_length, target_length = compute_input_and_target_lengths(
            inputs_length=data_args.max_seq_length,
            noise_density=data_args.mlm_prob,
            mean_noise_span_length=data_args.mean_noise_span,
        )
        collator = MixedT5DataCollatorForSpanCorruption(
            tokenizer=tokenizer,
            noise_density=data_args.mlm_prob,
            mean_noise_span_length=data_args.mean_noise_span,
            expandend_inputs_length=expanded_inputs_length,
            input_length=data_args.max_seq_length,
            target_length=target_length,  # for T5
            pad_token_id=tokenizer.pad_token_id,
            decoder_start_token_id=decoder_start_token_id,
            clm_ratio=data_args.clm_ratio,
            clm_max_output_length=data_args.clm_max_output_length,
            clm_max_doc=data_args.clm_max_doc,
        )

    elif data_args.collator_output_targets == "mixed_t5_clm":
        raise NotImplementedError

    else:
        raise NotImplementedError()

    return collator
