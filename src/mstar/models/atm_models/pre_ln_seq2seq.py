import math
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
import random

from typing import Any, Dict


from transformers.file_utils import ModelOutput

from transformers.activations import ACT2FN

import torch.nn.functional as F
from typing import Optional, Tuple

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)

from transformers.modeling_utils import (
    PreTrainedModel,
)
from transformers.utils import logging

from .pre_ln import PreLnEncoder, PreLnLayerNorm
from .mem_efficient_pre_ln import MemEfficientPreLnEncoder
from .atm_seq2seq import ATMSeq2SeqAttention, ATMSeq2SeqLearnedPositionalEmbedding

from .atm_seq2seq import shift_tokens_right, _make_causal_mask, _expand_mask

from .pre_ln_seq2seq_config import PreLNSeq2SeqConfig

logger = logging.get_logger(__name__)
# TODO: set level to INFO.


class PreLnSeq2SeqDecoderLayer(nn.Module):
    def __init__(self, config: PreLNSeq2SeqConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = ATMSeq2SeqAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            fp32_cast_query_key=config.fp32_cast_query_key,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = PreLnLayerNorm(config.d_model, eps=config.layer_norm_eps,
                                                   fp32_cast_layer_norm=config.fp32_cast_layer_norm)
        self.encoder_attn = ATMSeq2SeqAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            fp32_cast_query_key=config.fp32_cast_query_key,
        )
        self.encoder_attn_layer_norm = PreLnLayerNorm(config.d_model, eps=config.layer_norm_eps,
                                                      fp32_cast_layer_norm=config.fp32_cast_layer_norm)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = PreLnLayerNorm(config.d_model, eps=config.layer_norm_eps,
                                               fp32_cast_layer_norm=config.fp32_cast_layer_norm)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        encoder_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape `(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (:obj:`torch.FloatTensor`): cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (:obj:`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(config.encoder_attention_heads,)`.
            encoder_layer_head_mask (:obj:`torch.FloatTensor`): mask for encoder attention heads in a given layer of
                size `(config.encoder_attention_heads,)`.
            past_key_value (:obj:`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        # moving this up to become preLN
        # hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)
            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=encoder_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            # moving this up to become preLN
            # hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        # moving this up to make it preLN
        # hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class PreLNSeq2SeqPretrainedModel(PreTrainedModel):
    config_class = PreLNSeq2SeqConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


class PreLnSeq2SeqDecoder(PreLNSeq2SeqPretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a :class:`ATMSeq2SeqDecoderLayer`

    Args:
        config: PreLNSeq2SeqConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: PreLNSeq2SeqConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = ATMSeq2SeqLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model
        )
        self.layers = nn.ModuleList([PreLnSeq2SeqDecoderLayer(config) for _ in range(config.decoder_layers)])
        # remove this in preLN
        # self.layernorm_embedding = nn.LayerNorm(config.d_model)
        # add this for preLN
        self.layernorm_output = PreLnLayerNorm(config.d_model, eps=config.layer_norm_eps,
                                               fp32_cast_layer_norm=config.fp32_cast_layer_norm)

        self.model_parallel_devices = 0
        self.num_layers = config.decoder_layers

        self.init_weights()

    @property
    def starting_device(self):
        return self.embed_tokens.weight.device

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def set_model_parallel_devices(self, num_devices):
        self.model_parallel_devices = num_devices
 
    def parallelize(self, num_devices, start_device, use_cpu):
        assert num_devices > 1, "number of devices should be larger than 1"
        if use_cpu:
            device_list = ['cpu'] + ['cuda:{}'.format(i) for i in range(start_device, num_devices + start_device - 1)]
        else:
            device_list = ['cuda:{}'.format(i) for i in range(start_device, num_devices + start_device)]
 
        self.set_model_parallel_devices(num_devices)
 
        self.embed_tokens.to(device_list[-1])
        self.embed_positions.to(device_list[-1])
        self.layernorm_output.to(device_list[-1])
        k = self.num_layers/num_devices
        for i, layer in enumerate(self.layers):
            layer.to(device_list[int(i/k)])

    # pylint: disable=too-many-statements
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        encoder_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.


                `What are input IDs? <../glossary.html#input-ids>`__
            attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            encoder_hidden_states (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, encoder_sequence_length, hidden_size)`, `optional`):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, encoder_sequence_length)`, `optional`):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in ``[0, 1]``:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                `What are attention masks? <../glossary.html#attention-mask>`__
            head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            encoder_head_mask (:obj:`torch.Tensor` of shape :obj:`(num_layers, num_heads)`, `optional`):
                Mask to nullify selected heads of the attention modules in encoder to avoid performing cross-attention
                on hidden heads. Mask values selected in ``[0, 1]``:

                - 1 indicates the head is **not masked**,
                - 0 indicates the heas is **masked**.

            past_key_values (:obj:`Tuple[Tuple[torch.Tensor]]` of length :obj:`config.n_layers` with each tuple having 2 tuples each of which has 2 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden-states of the attention blocks. Can be used to speed up
                decoding.

                If :obj:`past_key_values` are used, the user can optionally input only the last
                :obj:`decoder_input_ids` (those that don't have their past key value states given to this model) of
                shape :obj:`(batch_size, 1)` instead of all :obj:`decoder_input_ids`` of shape :obj:`(batch_size,
                sequence_length)`.
            inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                into associated vectors than the model's internal embedding lookup matrix.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
            output_hidden_states (:obj:`bool`, `optional`):
                Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                for more detail.
            return_dict (:obj:`bool`, `optional`):
                Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
        """
        #### debug
        # if encoder_hidden_states.isinf().any().item():
        #     print("HS from encoder is inf")
        # elif encoder_hidden_states.isnan().any().item():
        #     print("HS from encoder is nan")
        ##########
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        # remove this in preLN
        # hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layers)
            ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if self.model_parallel_devices:
                if hidden_states.device != decoder_layer.device:
                    hidden_states = hidden_states.to(decoder_layer.device)
                    encoder_hidden_states = encoder_hidden_states.to(decoder_layer.device)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(decoder_layer.device)
                    if encoder_attention_mask is not None:
                        encoder_attention_mask = encoder_attention_mask.to(decoder_layer.device)
                    if output_hidden_states:
                        all_hidden_states = hidden_states.to(decoder_layer.device)

            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if getattr(self.config, "gradient_checkpointing", False) and self.training:

                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with `config.gradient_checkpointing=True`. Setting "
                        "`use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    encoder_head_mask[idx] if encoder_head_mask is not None else None,
                    None,
                )
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    encoder_layer_head_mask=(encoder_head_mask[idx] if encoder_head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add this for preLN
        hidden_states = self.layernorm_output(hidden_states)
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class PreLnEncoderForSeq2Seq(PreLNSeq2SeqPretrainedModel):
    # pylint: disable=unused-argument
    def __init__(self, config, token_embeddings, is_decoder=False):
        super().__init__(config)
        self.config = config
        #self.config.is_decoder = is_decoder
        # if is_decoder:
        #     self.config.add_cross_attention = True
        #self.is_decoder = is_decoder
        self.token_embeddings = token_embeddings  # will be passed and it is shared
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        #if not self.is_decoder:
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        # this for embeddings following PreLnEmbeddings
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # layer norm should also be added here
        self.mem_efficient_encoder = config.mem_efficient_encoder
        if not self.mem_efficient_encoder:
            self.pre_ln_encoder = PreLnEncoder(self.config)
            self.embedding_layer_norm = PreLnLayerNorm(config.d_model, eps=config.layer_norm_eps,
                                                       fp32_cast_layer_norm=config.fp32_cast_layer_norm)
        else:
            self.pre_ln_encoder = MemEfficientPreLnEncoder(self.config)
        self.use_attn_fuse_enc = config.use_attn_fuse_enc

        self.model_parallel_devices = 0
        self.num_layers = config.encoder_layers
        self.init_weights()

    def set_model_parallel_devices(self, num_devices):
        self.model_parallel_devices = num_devices
        if not self.mem_efficient_encoder:
            raise Exception('model parallel only supports mem_efficient_encoder')
 
    def parallelize(self, num_devices, start_device, use_cpu):
        if use_cpu:
            device_list = ['cpu'] + ['cuda:{}'.format(i) for i in range(start_device, num_devices + start_device - 1)]
        else:
            device_list = ['cuda:{}'.format(i) for i in range(start_device, num_devices + start_device)]
 
        self.set_model_parallel_devices(num_devices)
 
        self.position_embeddings.to(device_list[0])
        self.token_type_embeddings.to(device_list[0])
        self.position_ids = self.position_ids.to(device_list[0])
        self.pre_ln_encoder.parallelize(num_devices, start_device, use_cpu)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):

        input_shape = input_ids.size()

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        inputs_embeds = self.token_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.dropout(embeddings)
        # this is for using mem efficient encoder
        if not self.mem_efficient_encoder:
            hidden_states = self.embedding_layer_norm(embedding_output)
        else:
            hidden_states = embedding_output
        # if using attention fusion on the encoder then hidden states from all layers need to be returned
        if self.use_attn_fuse_enc:
            encoder_output_hidden_states = True
        else:
            encoder_output_hidden_states = output_hidden_states

        # this is for using mem efficient encoder
        if not self.mem_efficient_encoder:
            encoder_outputs = self.pre_ln_encoder(
                hidden_states,
                embedding_output,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=encoder_output_hidden_states,
                return_dict=return_dict,
            )
        else:
            encoder_outputs = self.pre_ln_encoder(
                hidden_states,
                attention_mask=extended_attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=encoder_output_hidden_states,
                return_dict=return_dict,
            )
        return encoder_outputs


class PreLNSeq2SeqModel(PreLNSeq2SeqPretrainedModel):
    def __init__(self, config: PreLNSeq2SeqConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.hidden_size, padding_idx)

        self.encoder = PreLnEncoderForSeq2Seq(config, self.shared)

        if config.share_embeddings:
            self.decoder = PreLnSeq2SeqDecoder(config, self.shared)
        else:
            self.decoder = PreLnSeq2SeqDecoder(config)
        self.use_attn_fuse_enc = config.use_attn_fuse_enc
        if self.use_attn_fuse_enc:
            self.task_vector = nn.Linear(config.hidden_size, 1, bias=False)

        self.model_parallel_devices = 0

        self.init_weights()

    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def set_model_parallel_devices(self, num_devices):
        self.model_parallel_devices = num_devices

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def parallelize(self, num_devices, start_device, use_cpu):
        assert num_devices % 2 == 0, "parallelize only support parallelizing into even number of GPUs"
        assert num_devices + start_device <= torch.cuda.device_count(), "number of available GPUs are less than " \
                                                                        "num_devices plus start_device for " \
                                                                        "model parallel"
        if use_cpu:
            device_list = ['cpu'] + ['cuda:{}'.format(i) for i in range(start_device, num_devices + start_device - 1)]
        else:
            device_list = ['cuda:{}'.format(i) for i in range(start_device, num_devices + start_device)]
        self.set_model_parallel_devices(num_devices)
        if num_devices == 2:
            self.shared.to(device_list[0])
            self.encoder.to(device_list[0])
            self.decoder.to(device_list[1])
        elif num_devices > 2:
            self.shared.to(device_list[0])
            self.encoder.parallelize(num_devices//2, start_device, use_cpu)
            self.decoder.parallelize(num_devices//2, start_device + num_devices//2, use_cpu)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # different to other models, PreLNSeq2Seq automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        # elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        #     encoder_outputs = BaseModelOutput(
        #         last_hidden_state=encoder_outputs[0],
        #         hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
        #         attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        #     )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        if self.use_attn_fuse_enc:
            # TODO: this can only be run once during generation (currently it is run everytime)
            all_hidden_outputs = torch.cat([hidden.unsqueeze(0) for hidden in encoder_outputs.hidden_states], dim=0)
            layers, batch, length, dim = all_hidden_outputs.shape
            task_dot = self.task_vector(all_hidden_outputs).reshape(all_hidden_outputs.shape[:-1])  # ~ (layers, batch, length)
            task_softmax = task_dot.softmax(dim=0).permute(1, 2, 0)  # ~ (batch, length, layers)
            all_hidden_outputs = all_hidden_outputs.permute(1, 2, 0, 3)  # ~ (batch, length, layers, dim)
            encoder_output_to_decoder = torch.bmm(task_softmax.reshape(-1, layers).unsqueeze(1),
                                                  all_hidden_outputs.reshape(-1, layers, dim)) \
                                                  .reshape(batch, length, dim)  # ~ (batch, length, dim)
        else:
            encoder_output_to_decoder = encoder_outputs.last_hidden_state

        if self.model_parallel_devices > 0:
            encoder_output_to_decoder = encoder_output_to_decoder.to(self.decoder.device)
            decoder_input_ids = decoder_input_ids.to(self.decoder.device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.device)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_output_to_decoder,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            encoder_head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
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


class PreLNSeq2SeqForConditionalGeneration(PreLNSeq2SeqPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [
        r"final_logits_bias",
        r"encoder\.version",
        r"decoder\.version",
        r"lm_head\.weight",
    ]

    def __init__(self, config: PreLNSeq2SeqConfig):
        super().__init__(config)
        self.model = PreLNSeq2SeqModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.hidden_size, self.model.shared.num_embeddings, bias=False)

        self.model_parallel_devices = 0
        self.init_weights()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    # pylint: disable=signature-differs
    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_model_parallel_devices(self, num_devices):
        self.model_parallel_devices = num_devices
 
    def parallelize(self, num_devices=2, start_device=0, use_cpu=False):
        """
        This function put the model on CPU and multiple GPUs allowing to run inference with larger model sizes on devices
        with variable memory. It supports partitioining the model into up to 8 devices.
        Since it is for a seq2seq model, it requires the number of devices to be even so encoder and decoder get
        partitioned into equal number of devices.
        :param num_devices: number of devices to partition the model into
        :param start_device: the starting device id of GPU
        :param use_cpu: partition the model on CPU
        :return:
        """
        logger.warn("Model parallelize is tested only for AlexaTM 20B Model.")
        assert num_devices % 2 == 0, "parallelize only support parallelizing into even number of GPUs"
        assert num_devices + start_device <= torch.cuda.device_count(), "number of available GPUs are less than " \
                                                                        "num_devices plus start_device for " \
                                                                        "model parallel"
        if use_cpu:
            device_list = ['cpu'] + ['cuda:{}'.format(i) for i in range(start_device, num_devices + start_device - 1)]
        else:
            device_list = ['cuda:{}'.format(i) for i in range(start_device, num_devices + start_device)]
 
        self.set_model_parallel_devices(num_devices)
 
        self.lm_head.to(device_list[-1])
        self.final_logits_bias = self.final_logits_bias.to(device_list[-1])
 
        self.model.parallelize(num_devices, start_device, use_cpu)
 
    def deparallelize(self):
        self.set_model_parallel_devices(0)
        self.model.to('cpu')
        self.lm_head.to('cpu')
        self.final_logits_bias = self.final_logits_bias.to('cpu')
        torch.cuda.empty_cache()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs.last_hidden_state) + self.final_logits_bias

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

#    @staticmethod
    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            if self.model_parallel_devices > 2 and beam_idx.device != layer_past[0].device:
                beam_idx = beam_idx.to(layer_past[0].device)
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    # have to rewrite this to work with model_parallel
    def _prepare_decoder_input_ids_for_generation(
        self, batch_size: int, decoder_start_token_id: int = None, bos_token_id: int = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None, device: torch.device = None,
    ) -> torch.LongTensor:
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            return model_kwargs.pop("decoder_input_ids")
        else:
            decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
            ## Force the device to use decoder.starting_device
            device = self.model.decoder.starting_device
            decoder_input_ids = (
                torch.ones((batch_size, 1), dtype=torch.long,
                        device=device)
                * decoder_start_token_id
            )
            return decoder_input_ids

# need to overwrite this for attention fusion
    # @staticmethod
    def _expand_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: torch.LongTensor = None,
        encoder_outputs: ModelOutput = None,
        **model_kwargs
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)
        if self.model_parallel_devices:
            attention_mask = attention_mask.to(self.model.decoder.starting_device)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            if self.model_parallel_devices:
                encoder_outputs["last_hidden_state"] = encoder_outputs["last_hidden_state"].to(
                    self.model.decoder.starting_device)
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx
            )
            # this is the additional change needed for attention fusion
            if encoder_outputs.hidden_states is not None:
                if self.model_parallel_devices:
                    encoder_outputs["hidden_states"] = encoder_outputs["hidden_states"].to(
                        self.model.decoder.starting_device)
                encoder_outputs["hidden_states"] = tuple(h.index_select(
                    0, expanded_return_idx
                ) for h in encoder_outputs.hidden_states)
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs
