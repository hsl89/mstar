""" T5 model modified from https://github.com/huggingface/transformers/blob/b424f0b4a301abcbf3c282114159371ee44c3e01/src/transformers/models/t5/modeling_t5.py."""

import copy
import math
import warnings
from typing import Optional, Tuple, Union

import torch
from torch.nn import CrossEntropyLoss
from torch import nn
from torch.utils.checkpoint import checkpoint

from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    is_torch_fx_proxy,
    logging,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers import T5Config, PretrainedConfig, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import (
    T5LayerFF,
    T5DenseActDense,
    T5DenseGatedActDense,
    T5EncoderModel,
    T5LayerNorm,
    T5LayerSelfAttention,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)


class MStarT5Config(PretrainedConfig):
    r"""
    Extends the transformers T5 Config with additional options such as fused kernels
    Arguments:
        vocab_size (`int`, *optional*, defaults to 32128):
            Vocabulary size of the T5 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`T5Model`] or [`TFT5Model`].
        d_model (`int`, *optional*, defaults to 512):
            Size of the encoder layers and the pooler layer.
        d_kv (`int`, *optional*, defaults to 64):
            Size of the key, query, value projections per attention head. `d_kv` has to be equal to `d_model //
            num_heads`.
        d_ff (`int`, *optional*, defaults to 2048):
            Size of the intermediate feed forward layer in each `T5Block`.
        num_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        num_decoder_layers (`int`, *optional*):
            Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
        num_heads (`int`, *optional*, defaults to 8):
            Number of attention heads for each attention layer in the Transformer encoder.
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            The number of buckets to use for each attention layer.
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            The maximum distance of the longer sequences for the bucket separation.
        dropout_rate (`float`, *optional*, defaults to 0.1):
            The ratio for all dropout layers.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        feed_forward_proj (`string`, *optional*, defaults to `"relu"`):
            Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`. T5v1.1 uses the
            `"gated-gelu"` feed forward projection. Original T5 uses `"relu"`.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        use_fused_attention (`bool`, *optional*, defaults to `True`):
            Uses custom, non-huggingface, attention function
        softmax_type (`string`, *optional*, defaults to `mstar_fused`):
             mstar_fused  uses megatron softmax, can use `torch`. Only takes effect when `use_fused_attention` is True):
    """
    model_type = "mstar-t5"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        vocab_size=32128,
        d_model=512,
        d_kv=64,
        d_ff=2048,
        num_layers=6,
        num_decoder_layers=None,
        num_heads=8,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        use_fused_attention=True,
        softmax_precision="fp16",
        softmax_type="mstar_fused",
        positional_embedding="t5",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache
        self.use_fused_attention = use_fused_attention
        self.softmax_precision = softmax_precision
        self.softmax_type = softmax_type
        assert positional_embedding in [
            "t5",
            "alibi",
        ], f"Positional embedding {positional_embedding} not supported"
        self.positional_embedding = positional_embedding

        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer."
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )

        # for backwards compatibility
        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )


logger = logging.get_logger(__name__)


def get_slopes(n):
    """Utility function for alibi"""

    def get_slopes_power_of_2(n):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        # when the number of heads is not a power of 2, we use this workaround
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


def get_alibi(maxpos, attn_heads):
    """
    Creates alibi positional embeddings
    Following official encoder-decoder
    https://github.com/ofirpress/attention_with_linear_biases/issues/5 
    """

    context_position = torch.arange(maxpos)[:, None]
    memory_position = torch.arange(maxpos)[None, :]
    relative_position = memory_position - context_position
    relative_position = (
        torch.abs(relative_position).unsqueeze(0).expand(attn_heads, -1, -1)
    )
    slopes = torch.Tensor(get_slopes(attn_heads)) * -1.0
    new_alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position
    return new_alibi


def attention_mask_func(attention_scores, attention_mask):
    """Helper function for megatron softmax"""
    attention_mask = attention_mask.bool()
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


try:
    from apex.normalization import FusedRMSNorm

    T5LayerNorm = FusedRMSNorm  # noqa

    logger.info(
        "Discovered apex.normalization.FusedRMSNorm - will use it instead of T5LayerNorm"
    )

except ImportError:
    # using the normal T5LayerNorm
    pass
except Exception:
    logger.warning("discovered apex but it failed to load, falling back to T5LayerNorm")


class FusedT5Attention(nn.Module):
    def __init__(
        self,
        config: T5Config,
        has_relative_attention_bias=False,
        is_cross_attention=False,
    ):
        super().__init__()
        self.config = config
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.is_cross_attention = is_cross_attention

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            if self.config.positional_embedding == "alibi":
                # re-create buffer later if this is too small
                ALIBI_INIT_SZ = 2048
                if config.softmax_precision == "fp16":
                    self.alibi_dtype = torch.float16
                elif config.softmax_precision == "bf16":
                    self.alibi_dtype = torch.bfloat16
                else:
                    logger.warning("Alibi mask in float32 is slow")
                    self.alibi_dtype = torch.float32

                self.create_alibi_bias(ALIBI_INIT_SZ, ALIBI_INIT_SZ, self.alibi_dtype)

            elif self.config.positional_embedding == "t5":
                self.relative_attention_bias = nn.Embedding(
                    self.relative_attention_num_buckets, self.n_heads
                )
            else:
                raise NotImplementedError(
                    f"Embedding type {self.config.positional_embedding} not implemented"
                )

        self.pruned_heads = set()
        self.gradient_checkpointing = False

        if self.config.softmax_type == "mstar_fused":
            from mstar.megatron.fused_softmax import FusedScaleMaskSoftmax, AttnMaskType

            fp16, bf16 = (
                config.softmax_precision == "fp16",
                config.softmax_precision == "bf16",
            )

            if self.is_decoder and not self.is_cross_attention:
                attn_mask_type = AttnMaskType.causal
            else:
                attn_mask_type = AttnMaskType.padding

            self.scale_mask_softmax = FusedScaleMaskSoftmax(
                input_in_fp16=fp16,
                input_in_bf16=bf16,
                attn_mask_type=attn_mask_type,
                scaled_masked_softmax_fusion=True,
                mask_func=attention_mask_func,
                softmax_in_fp32=True,
                scale=None,
            )
        else:
            assert (
                self.config.softmax_type == "torch"
            ), "Softmax type {} not supported".format(self.config.softmax_type)

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def create_alibi_bias(self, query_length, key_length, dtype):
        assert (
            query_length == key_length
        ), f"Query {query_length} key {key_length} What to do for cross-attention?"
        alibi_positional_bias = get_alibi(
            query_length, self.config.num_heads
        ).unsqueeze(0)
        if hasattr(self, "alibi_positional_bias"):
            logger.info("Recreating alibi positional bias on same device")
            alibi_positional_bias = alibi_positional_bias.to(
                self.alibi_positional_bias.device
            )

        self.register_buffer("alibi_positional_bias", alibi_positional_bias.to(dtype))

    def compute_alibi_bias(self, query_length, key_length):
        assert (
            query_length == key_length
        ), f"Query {query_length} key {key_length} What to do for cross-attention?"

        if self.alibi_positional_bias.shape[2] >= query_length:
            return self.alibi_positional_bias[..., :query_length, :query_length]
        else:
            logger.info(
                "Recreating alibi positional bias to meet longer sequence length"
            )
            self.create_alibi_bias(query_length, key_length, self.alibi_dtype)

            return self.alibi_positional_bias[..., :query_length, :query_length]

    def compute_t5_bias(self, query_length, key_length, device=None):
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[
            :, None
        ]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[
            None, :
        ]
        relative_position = (
            memory_position - context_position
        )  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, query_length, key_length)
        return values

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if self.config.positional_embedding == "t5":
            positional_bias = self.compute_t5_bias(query_length, key_length)

        elif self.config.positional_embedding == "alibi":
            positional_bias = self.compute_alibi_bias(query_length, key_length)

        else:
            raise NotImplementedError(
                f"Positional embedding {self.config.positional_embedding}"
            )

        return positional_bias.to(device)

    # preserve original huggingface code, ignore pylint warning
    # pylint: disable=too-many-statements
    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += (
                past_key_value[0].shape[2] if query_length is None else query_length
            )

        key_length = (
            real_seq_length if key_value_states is None else key_value_states.shape[1]
        )

        def shape(states):
            """projection"""
            return states.view(
                batch_size, -1, self.n_heads, self.key_value_proj_dim
            ).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return (
                states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
            )

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length),
                    device=self.q.weight.device,
                    dtype=self.q.weight.dtype,
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=self.q.weight.device
                )

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None and self.config.softmax_type != "mstar_fused":
                # if we use fused softmax the mask is part of the softmax so we apply it later
                position_bias = (
                    position_bias + mask
                )  # (batch_size, n_heads, seq_length, key_length)

        # get query states
        query_states = shape(
            self.q(hidden_states)
        )  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states,
            self.k,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states = project(
            hidden_states,
            self.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        scores += position_bias

        if self.config.softmax_type == "mstar_fused":
            batch_size, num_heads = query_states.size(0), query_states.size(1)
            query_length, key_length = query_states.size(2), key_states.size(2)
            output_size = (batch_size, num_heads, query_length, key_length)

            if not self.scale_mask_softmax.is_kernel_available(*output_size):
                logger.warning(
                    "Selected fused softmax but it is not available for given output size\n"
                    "Falling back to slower torch softmax"
                )

            if mask.shape[2] != query_length:
                # kernel requires mask to be appropriately sized
                # some token masks are just key masks
                # mask=None
                assert (
                    mask.shape[2] == 1 and len(mask.shape) == 4
                ), "Fused softmax does not support per-head mask"
                mask = mask.repeat(1, 1, query_length, 1)

            # scale_mask_softmax casts mask to boolean
            if (
                self.scale_mask_softmax.is_kernel_available(*output_size)
                and scores.dtype == torch.float32
            ):
                # this will try to run fp32 and fail in fused softmax unless we cast
                if self.config.softmax_precision == "fp16":
                    attn_weights = self.scale_mask_softmax(
                        scores.to(torch.float16), mask != 0
                    ).type_as(scores)
                elif self.config.softmax_precision == "bf16":
                    attn_weights = self.scale_mask_softmax(
                        scores.to(torch.bfloat16), mask != 0
                    ).type_as(scores)
                else:
                    raise NotImplementedError()
            else:
                attn_weights = self.scale_mask_softmax(scores, mask != 0).type_as(
                    scores
                )

        elif self.config.softmax_type == "torch":
            attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        else:
            raise NotImplementedError(
                "Softmax type {} not supported".format(self.config.softmax_type)
            )
        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(
            torch.matmul(attn_weights, value_states)
        )  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (
            (key_states, value_states) if (self.is_decoder and use_cache) else None
        )
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class MStarT5LayerSelfAttention(T5LayerSelfAttention):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        if config.use_fused_attention:
            # all attention modules formed inside this module are standard, not cross, attention
            self.SelfAttention = FusedT5Attention(
                config, has_relative_attention_bias=has_relative_attention_bias
            )
        else:
            self.SelfAttention = T5Attention(
                config, has_relative_attention_bias=has_relative_attention_bias
            )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)


class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )
        self.pruned_heads = set()
        self.gradient_checkpointing = False

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=128
    ):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[
            :, None
        ]
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[
            None, :
        ]
        relative_position = (
            memory_position - context_position
        )  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(
            relative_position_bucket
        )  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(
            0
        )  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += (
                past_key_value[0].shape[2] if query_length is None else query_length
            )

        key_length = (
            real_seq_length if key_value_states is None else key_value_states.shape[1]
        )

        def shape(states):
            """projection"""
            return states.view(
                batch_size, -1, self.n_heads, self.key_value_proj_dim
            ).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return (
                states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
            )

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(
            self.q(hidden_states)
        )  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states,
            self.k,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states = project(
            hidden_states,
            self.v,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=scores.device
                )

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            if mask is not None:
                position_bias = (
                    position_bias + mask
                )  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(
            torch.matmul(attn_weights, value_states)
        )  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        present_key_value_state = (
            (key_states, value_states) if (self.is_decoder and use_cache) else None
        )
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.use_fused_attention:
            # all attention modules formed inside this module are standard, not cross, attention
            self.EncDecAttention = FusedT5Attention(
                config, has_relative_attention_bias=False, is_cross_attention=True
            )
        else:
            self.EncDecAttention = T5Attention(
                config, has_relative_attention_bias=False
            )

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[
            1:
        ]  # add attentions if we output them
        return outputs


class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(
            MStarT5LayerSelfAttention(
                config, has_relative_attention_bias=has_relative_attention_bias
            )
        )
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

    # preserve original huggingface code, ignore pylint warning
    # pylint: disable=unused-argument
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):

        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning(
                    "`past_key_values` is passed to the encoder. Please make sure this is intended."
                )
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[
            2:
        ]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if (
                hidden_states.dtype == torch.float16
                and torch.isinf(hidden_states).any()
            ):
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(
                    hidden_states, min=-clamp_value, max=clamp_value
                )

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = (
                    present_key_value_state + cross_attention_outputs[1]
                )

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return (
            outputs
        )  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


class T5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = T5Config
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["T5Block"]

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = (
            self.config.initializer_factor
        )  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (T5Model, T5ForConditionalGeneration, T5EncoderModel)):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
            if hasattr(module, "lm_head") and not self.config.tie_word_embeddings:
                module.lm_head.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, T5DenseActDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_model) ** -0.5)
            )
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_ff) ** -0.5)
            )
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedActDense):
            module.wi_0.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_model) ** -0.5)
            )
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_model) ** -0.5)
            )
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(
                mean=0.0, std=factor * ((self.config.d_ff) ** -0.5)
            )
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5Attention) or isinstance(module, FusedT5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads

            module.q.weight.data.normal_(
                mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5)
            )
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))

            module.o.weight.data.normal_(
                mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5)
            )

            if module.has_relative_attention_bias:
                if module.config.positional_embedding == "alibi":
                    pass
                else:
                    module.relative_attention_bias.weight.data.normal_(
                        mean=0.0, std=factor * ((d_model) ** -0.5)
                    )

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (FusedT5Attention, T5Attention, T5Stack)):
            module.gradient_checkpointing = value

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert decoder_start_token_id is not None, (
            "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
            " See T5 docs for more information"
        )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(
                input_ids.shape[:-1] + (1,), decoder_start_token_id
            )
            shifted_input_ids = torch.cat(
                [shifted_input_ids, input_ids[..., :-1]], dim=-1
            )
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert (
            pad_token_id is not None
        ), "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(
            shifted_input_ids >= 0
        ).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [
                T5Block(config, has_relative_attention_bias=bool(i == 0))
                for i in range(config.num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = (
            "cpu"
            if "cpu" in self.device_map.keys()
            else "cuda:" + str(min(self.device_map.keys()))
        )
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        # preserve original huggingface code, ignore pylint warning
        # pylint: disable=consider-using-enumerate
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    # preserve original huggingface code, ignore pylint warning
    # pylint: disable=too-many-statements
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds"
            )

        if inputs_embeds is None:
            assert (
                self.embed_tokens is not None
            ), "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = (
            past_key_values[0][0].shape[2] + seq_length
            if past_key_values is not None
            else seq_length
        )

        if use_cache is True:
            assert (
                self.is_decoder
            ), f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(
                inputs_embeds.device
            )
        if (
            self.is_decoder
            and encoder_attention_mask is None
            and encoder_hidden_states is not None
        ):
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size,
                encoder_seq_length,
                device=inputs_embeds.device,
                dtype=torch.long,
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = (
                encoder_hidden_states.size()
            )
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device
                )
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(
            cross_attn_head_mask, self.config.num_layers
        )
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(
            zip(self.block, past_key_values)
        ):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(
                        hidden_states.device
                    )
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                        hidden_states.device
                    )
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(
                        hidden_states.device
                    )
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(
                        hidden_states.device
                    )
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[
                    4 if output_attentions else 3
                ]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (
                    present_key_value_state,
                )

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


T5_START_DOCSTRING = r"""

    The T5 model was proposed in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text
    Transformer](https://arxiv.org/abs/1910.10683) by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan
    Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It's an encoder decoder transformer pre-trained in a
    text-to-text denoising generative setting.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`T5Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

T5_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`T5Tokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            [What are input IDs?](../glossary#input-ids)

            To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using [`T5Tokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are decoder input IDs?](../glossary#decoder-input-ids)

            T5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values`
            is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`).

            To know more on how to prepare `decoder_input_ids` for pretraining take a look at [T5
            Training](./t5#training).
        decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
            Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
            be used by default.
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in `[0,
            1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        cross_attn_head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected in
                `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

        encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
            Tuple consists of (`last_hidden_state`, `optional`: *hidden_states*, `optional`: *attentions*)
            `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states at
            the output of the last layer of the encoder. Used in the cross-attention of the decoder.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
            representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be
            input (see `past_key_values`). This is useful if you want more control over how to convert
            `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value
            of `inputs_embeds`.

        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

T5_ENCODER_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using [`T5Tokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for detail.

            To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


@add_start_docstrings(
    "The bare T5 Model transformer outputting raw hidden-states without any specific head on top.",
    T5_START_DOCSTRING,
)
class T5Model(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import T5Tokenizer, T5Model

        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5Model.from_pretrained("t5-small")

        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

        >>> # forward pass
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


@add_start_docstrings(
    """T5 Model with a `language modeling` head on top.""", T5_START_DOCSTRING
)
class MStarT5ForConditionalGeneration(T5PreTrainedModel):
    config_class = MStarT5Config
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
        r"lm_head.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
        r"encoder.block.0.layer.0.SelfAttention.alibi_positional_bias",  # alibi bias buffer does not exist at initialization, will be re-created
        r"decoder.block.0.layer.0.SelfAttention.alibi_positional_bias",  # alibi bias buffer does not exist at initialization, will be re-created
    ]

    def __init__(self, config: MStarT5Config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning(
                "You might want to consider setting `use_cache=True` to speed up decoding"
            )
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(
                        0, beam_idx.to(layer_past_state.device)
                    ),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (
                reordered_layer_past_states,
            )
        return reordered_decoder_past
