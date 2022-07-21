# Compute the FLOPs for decoder-only and encoder-decoder text models
from typing import Optional


def attention_block_formula(hidden_size, query_seq_len, key_seq_len):
    """
    Computes FLOPs for attention block, includes linear projection
    For self-attention query_seq_len=key_seq_len
    """
    c_query = 2 * key_seq_len * (hidden_size**2)
    # k,v must have same seq length
    c_key = 2 * key_seq_len * (hidden_size**2)
    c_val = c_key
    c_qkv = c_query + c_key + c_val

    # attention_matrix
    c_att = 2 * query_seq_len * key_seq_len * hidden_size
    # attention matrix times values, same dimensions as above, but permuted
    c_av = c_att

    # linear projection
    c_proj = 2 * query_seq_len * (hidden_size**2)

    return c_qkv + c_av + c_att + c_proj


def megatron_lm_formula(
    activation_checkpointing: bool,
    vocab_size: int,
    hidden_size: int,
    num_layers: int,
    seq_len: int,
    include_softmax: bool,
    use_gated_mlp: bool,
):
    """Calculate TFLOPs using https://cs.stanford.edu/~matei/papers/2021/sc_megatron_lm.pdf formulas
    Applies to either encoder-only or decoder-only models, but not cross-attention
    """

    # self-attention
    c_self_attention = attention_block_formula(
        hidden_size=hidden_size, query_seq_len=seq_len, key_seq_len=seq_len
    )

    # feedforward cost, assumes 4x expansion
    c_ff = 16 * seq_len * (hidden_size**2)
    # gated MLP adds one matmul and pointwise multiplication
    if use_gated_mlp:
        c_ff += 8 * seq_len * (hidden_size**2) + seq_len * 4 * hidden_size

    c_layer = c_self_attention + c_ff

    # eveything except softmax has +1x fwd +1x recompute
    # if activation checkpointing  and +2x for bwd
    # softmax just +1x for fwd and +2x for bwd
    c_fwd_bwd = 4.0 if activation_checkpointing else 3.0

    fwd_bwd_flops = c_fwd_bwd * num_layers * c_layer

    if include_softmax:
        # logit cost
        c_logit = 2 * seq_len * hidden_size * vocab_size
        fwd_bwd_flops += 3 * c_logit

    tflop_scaling = 1e12

    return fwd_bwd_flops / tflop_scaling


def encoder_decoder_formula(
    activation_checkpointing: bool,
    vocab_size: int,
    hidden_size: int,
    decoder_num_layers: int,
    decoder_seq_len: int,
    encoder_seq_len: int,
    use_gated_mlp: bool,
):
    """Compute encoder-decoder decoder tflops. Main difference to megatron_lm formula is different input/output seq_len"""
    # self-attention
    c_self_attention = attention_block_formula(
        hidden_size=hidden_size,
        query_seq_len=decoder_seq_len,
        key_seq_len=decoder_seq_len,
    )
    # self-attention
    c_cross_attention = attention_block_formula(
        hidden_size=hidden_size,
        query_seq_len=decoder_seq_len,
        key_seq_len=encoder_seq_len,
    )

    # feedforward cost, assumes 4x expansion
    c_ff = 16 * decoder_seq_len * (hidden_size**2)
    # gated MLP adds one matmul and pointwise multiplication
    if use_gated_mlp:
        c_ff += (
            8 * decoder_seq_len * (hidden_size**2) + decoder_seq_len * 4 * hidden_size
        )

    c_layer = c_self_attention + c_cross_attention + c_ff

    # logit cost
    c_logit = 2 * decoder_seq_len * hidden_size * vocab_size

    # eveything except softmax has +1x fwd +1x recompute
    # if activation checkpointing  and +2x for bwd
    # softmax just +1x for fwd and +2x for bwd
    c_fwd_bwd = 4.0 if activation_checkpointing else 3.0

    fwd_bwd_flops = c_fwd_bwd * decoder_num_layers * c_layer + 3 * c_logit

    tflop_scaling = 1e12

    return fwd_bwd_flops / tflop_scaling


def compute_tflops_per_gpu(
    model_type: str,
    sec_per_step: int,
    activation_checkpointing: bool,
    vocab_size: int,
    hidden_size: int,
    decoder_num_layers: int,
    micro_batchsize: float,
    decoder_seq_len: int,
    encoder_seq_len: Optional[int] = 0,
    encoder_num_layers: Optional[int] = 0,
    use_gated_mlp: Optional[bool] = False,
):
    """Compute the TFLOPS of a given model
    This applies to encoder-decoder and decoder-only models during training with with teacher forcing.
    This function assumes that gradient checkpointing is used.

    Parameters
    __________
    model_type: encoder-decoder or encoder-only
    sec_per_step: how many seconds to run one fwd/bwd step
    activation_checkpointing: does training use activation checkpointing?
    vocab_size: number of intput/output embeddings (assumed to be equal)
    hidden_size: size of pre-attention input embeddings, in huggingface config style this is d_model not d_ff
    decoder_num_layers: number of layers that have input decoder_seq_len input
    micro_batchsize: per-gpu batch size, can be <1 with model parallelism
    decoder_seq_len: number of inputs to the decoder
    encoder_seq_len: number of inputs to the encoder, only required for encoder-decoder models
    encoder_num_layers: number of layers that have input length encoder_seq_len, only necessary for encoder-decoder
    use_gated_mlp: If using gated mlp, which adds one matrix multiply
    """

    assert model_type in ["decoder", "encoder_decoder"], "Model type not supported"

    if model_type == "decoder":
        tflops_per_example = megatron_lm_formula(
            activation_checkpointing=activation_checkpointing,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=decoder_num_layers,
            seq_len=decoder_seq_len,
            include_softmax=True,
            use_gated_mlp=use_gated_mlp,
        )

    elif model_type == "encoder_decoder":
        assert (
            encoder_seq_len > 0 and encoder_num_layers > 0
        ), "Must provide separate encoder and decoder info"
        # add encoder tflops, needs different calculation
        # because encoder cross-attention and possible different sequence
        # length
        encoder_tflops_per_example = megatron_lm_formula(
            activation_checkpointing=activation_checkpointing,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=encoder_num_layers,
            seq_len=encoder_seq_len,
            include_softmax=False,
            use_gated_mlp=use_gated_mlp,
        )
        # add encoder-decoder deocder tflops with cross attention
        decoder_tflops_per_example = encoder_decoder_formula(
            activation_checkpointing=activation_checkpointing,
            vocab_size=vocab_size,
            decoder_num_layers=decoder_num_layers,
            hidden_size=hidden_size,
            decoder_seq_len=decoder_seq_len,
            encoder_seq_len=encoder_seq_len,
            use_gated_mlp=use_gated_mlp,
        )

        tflops_per_example = encoder_tflops_per_example + decoder_tflops_per_example

    return tflops_per_example * micro_batchsize / sec_per_step
