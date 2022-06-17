import math
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import (
    RobertaLMHead,
    RobertaPooler,
)

logger = logging.get_logger(__name__)


class PreLnConfig(RobertaConfig):
    """For use with PreLnPreTrainedModel. Fork of RobertaConfig, plus settings for fp32 casting to improve stability.

    We have found that when training large models with fp16 precision, it is important to use fp32_cast_layer_norm=True
        and fp32_cast_query_key="baddbmm". (You may also use fp32_cast_query_key="manual", it's slightly slower
        and possibly slightly more stable, although in practice we have not seen much stability difference between
        baddbmm vs. manual here.)

    We have found that fp32_cast_attention_scores and fp32_cast_logits are usually not needed (since torch Softmax
        does accumulation in fp32 anyway), so we set them to default False. Keeping them here as options anyway
        to support existing models trained with these options, and also in case we want to try again in the future.

    Args:
        fp32_cast_layer_norm (bool, default=True): Cast to fp32 before computing layer norm.
        fp32_cast_query_key (str, default="baddbmm"): Cast to fp32 before query-key multiplication. Choices are either
            None (which means no casting), or "baddbmm", which uses 'torch.baddbmm', or "manual" which manually casts
            to fp32 then does 'torch.matmul'.
        fp32_cast_attention_scores (bool, default=False): Cast to fp32 before computing Softmax on attention scores.
        fp32_cast_logits (bool, default=False): Cast to fp32 before computing Softmax on logits.

    """
    model_type = "atm-PreLn"

    def __init__(
            self,
            fp32_cast_layer_norm=True,
            fp32_cast_query_key="baddbmm",
            fp32_cast_attention_scores=False,
            fp32_cast_logits=False,
            **kwargs):
        """Constructs PreLnConfig."""
        self.fp32_cast_layer_norm = fp32_cast_layer_norm
        self.fp32_cast_query_key = fp32_cast_query_key
        self.fp32_cast_attention_scores = fp32_cast_attention_scores
        self.fp32_cast_logits = fp32_cast_logits
        super().__init__(**kwargs)


class PreLnForMaskedLMConfig(PreLnConfig):
    model_type = "atm-PreLnForMaskedLM"


class PreLnEmbeddings(nn.Module):
    """
    Same as RobertaEmbedding but removing the LayerNorm.
    """

    # Copied from transformers.models.bert.modeling_roberta.RobertaEmbeddings.__init__
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        # Copied from transformers.models.bert.modeling_bert.BertEmbeddings.forward
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        # ~~~ leaving this here to remind ourselves of why we did this. We should remove later
        # REMOVED --> embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    # If the removal of the padding_idx works, this function can be removed
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)


# Copied from transformers.models.bert.modeling_bert.RobertaSelfAttention and added the layer norm at input
class PreLnSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type in ("relative_key", "relative_key_query"):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.fp32_cast_query_key = config.fp32_cast_query_key
        self.fp32_cast_attention_scores = config.fp32_cast_attention_scores

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        # b = batch size, e.g. 8
        # h = num attention heads, e.g. 32
        # l = seq len, e.g. 512
        # d = dim per head, e.g. 80 (for 80 * 32 = 2560 total hidden dim)
        # b, h, l, d
        query_layer = self.transpose_for_scores(mixed_query_layer)
        # b, h, l, d
        key_layer = self.transpose_for_scores(mixed_key_layer)
        # b, h, l, d
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # TODO, relative embedding stuff instead of empty.

        # Following with from Shuai Zheng (shzheng@) here:
        # https://code.amazon.com/packages/M5Transformers/commits/a6f6ee88562bbdbbd3192b4f3633d5ac81f707cd#
        # Also joint work here with waelhamz@, jgmf@, sterawls@, harakere@, chanprak@, jincao@, ssoltan@, khhaida@
        if self.fp32_cast_query_key == "baddbmm":
            # b*h, l, d
            # Switching from view to reshape following this error message:
            # RuntimeError: view size is not compatible with input tensor's size and stride
            # (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
            query_layer_shaped = query_layer.reshape(-1, query_layer.size(-2), query_layer.size(-1))
            # b*h, l, d
            key_layer_shaped = key_layer.reshape(-1, key_layer.size(-2), key_layer.size(-1))

            # b*h, l, l
            empty = torch.empty(
                [query_layer_shaped.size(0), query_layer_shaped.size(-2), key_layer_shaped.size(-2)],
                device=query_layer_shaped.device, dtype=query_layer_shaped.dtype)

            attention_scores = torch.baddbmm(
                empty, query_layer_shaped, key_layer_shaped.transpose(-1, -2),
                beta=0.0, alpha=1.0 / math.sqrt(self.attention_head_size))

            # b, h, l, l
            # Note: here we are using dimensions (0, 1, 2) for ease of implementation and readability.
            #   In future, it may be preferable to use negative values to specify the tensor dimensions,
            #   to be robust to situations where there are extra dimensions between the batch dimension,
            #   and the dimensions you are trying to identify.
            attention_scores = attention_scores.reshape(
                query_layer.size(0), query_layer.size(1), query_layer.size(2), key_layer.size(2))

        elif self.fp32_cast_query_key == "manual":
            attention_scores = torch.matmul(query_layer.float(), key_layer.transpose(-1, -2).float())
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        elif self.fp32_cast_query_key is None:
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        else:
            raise NotImplementedError(
                f"Unsupported fp32_cast_query_key: {self.fp32_cast_query_key}. "
                f"Choose from 'baddbmm' (recommended), 'manual', or None (meaning do not fp32 cast).")

        # TODO fix this.
        if self.position_embedding_type in ("relative_key", "relative_key_query"):
            logger.warning(
                "Relative position embeddings do not yet have implementation of fp32 casting. This fp32 casting "
                "was crucial for training stability with absolute positional embeddings. If you see unstable training, "
                "e.g. loss and logits going to nan, try implementing the fp32 casting tricks here.")
            seq_length = hidden_states.size()[1]
            # One is a row, one is a column
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            # this distance is now a matrix, columns and rows are
            # this output is l x l (l=seq_len)
            distance = position_ids_l - position_ids_r
            # Now l x l x d_head
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                # query is micro_batch x num_heads(32)? x len(seq_len)? x per_head_dimension
                # pos_emb is l(seq_len) x r x dimension
                # broadcast b ; transpose
                # l == r
                # TODO, could do this using baddbmm
                #
                # A @ B = [A + a] @ [B + b]
                #
                # A @ B + a @ B + A @ b + a @ b
                #
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                # attention scores are bhll
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # Option 1: cast to fp32.
        if self.fp32_cast_attention_scores:
            attention_scores = attention_scores.float()
        # Option 2: keep as (or cast back if applicable) to fp16.
        else:
            attention_scores = attention_scores.to(hidden_states.dtype)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # In case did cast, cast back.
        attention_probs = attention_probs.to(hidden_states.dtype)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class PreLnSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states + input_tensor


# Copied from transformers.models.bert.modeling_bert.BertAttention with Bert->Roberta
class PreLnAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.self = PreLnSelfAttention(config)
        self.output = PreLnSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        before_layer_norm_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], before_layer_norm_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_roberta.RobertaIntermediate (adding layer norm at input)
class PreLnIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # From HF t5 code
        # https://github.com/huggingface/transformers/blob/master/src/transformers/models/t5/modeling_t5.py#L243
        self.layer_norm = PreLnLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, fp32_cast_layer_norm=config.fp32_cast_layer_norm)
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_ removing layer norm
class PreLnOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states + input_tensor


class PreLnLayerNorm(nn.Module):
    def __init__(self, dimension, eps=1e-05, elementwise_affine=True, fp32_cast_layer_norm=True):
        super().__init__()
        assert isinstance(dimension, int)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.fp32_cast_layer_norm = fp32_cast_layer_norm

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dimension))
            self.bias = nn.Parameter(torch.zeros(dimension))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype

        if self.fp32_cast_layer_norm:
            hidden_states = hidden_states.float()

        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        mean = hidden_states.mean(dim=-1, keepdim=True)

        # now mean 0 and variance 1
        hidden_states = (hidden_states - mean) * torch.rsqrt(variance + self.eps)

        # In case did cast.
        hidden_states = hidden_states.to(orig_dtype)

        if self.elementwise_affine:
            hidden_states = hidden_states * self.weight + self.bias

        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLayer with Bert->Roberta
class PreLnLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = PreLnAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            assert self.is_decoder, f"{self} should be used as a decoder model if cross attention is added"
            logger.error("skip/layer norm code is not tested for cross attention. "
                         "We believe it is done right but not tested. If you test it and like it, please remove "
                         "this error message")
            self.query_layer_norm = PreLnLayerNorm(
                config.hidden_size, eps=config.layer_norm_eps, fp32_cast_layer_norm=config.fp32_cast_layer_norm)
            self.crossattention = PreLnAttention(config)
        self.intermediate = PreLnIntermediate(config)
        self.output = PreLnOutput(config)

        # bringing this out here to avoid the chunching business
        self.layer_norm = PreLnLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, fp32_cast_layer_norm=config.fp32_cast_layer_norm)

    def forward(
        self,
        hidden_states,
        before_layer_norm,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            before_layer_norm,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            assert hasattr(
                self, "crossattention"
            ), f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"

            # adding layer norm in the case of cross attention
            query_before_layer_norm = attention_output
            attention_output = self.query_layer_norm(attention_output)
            cross_attention_outputs = self.crossattention(
                attention_output,
                query_before_layer_norm,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        before_layer_norm = layer_output
        layer_output = self.layer_norm(layer_output)
        outputs = (layer_output, before_layer_norm) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->Roberta
class PreLnEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([PreLnLayer(config) for _ in range(config.num_hidden_layers)])

        if getattr(config, "gradient_checkpointing", False):
            logger.warning(
                "PreLnEncoder currently checkpoints both a layer-normalized and "
                "non layer-normalized version of its activations. this essentially "
                "doubles memory consumption by activations. see mem_efficient_pre_ln "
                "if you are experiencing memory pressure (or reduced performance "
                "resulting from memory pressure) when using this model."
            )

    def forward(
        self,
        hidden_states,
        before_layer_norm,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    before_layer_norm,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    before_layer_norm,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            before_layer_norm = layer_outputs[1]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[3],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class PreLnPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    base_model_prefix = "pre_ln_encoder"

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            scaled_segma = self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers)
            if (isinstance(module, PreLnSelfOutput) and hasattr(module, 'dense')) \
                    or (isinstance(module, PreLnOutput) and hasattr(module, 'dense')):
                module.dense.weight.data.normal_(mean=0.0, std=scaled_segma)
            else:
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (nn.LayerNorm, PreLnLayerNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class PreLnModel(PreLnPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need`_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.

    .. _`Attention is all you need`: https://arxiv.org/abs/1706.03762

    """

    config_class = PreLnConfig
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = PreLnEmbeddings(config)
        self.embedding_layer_norm = PreLnLayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, fp32_cast_layer_norm=config.fp32_cast_layer_norm)

        self.encoder = PreLnEncoder(config)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward
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
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``: ``1`` for
            tokens that are NOT MASKED, ``0`` for MASKED tokens.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

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

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        hidden_states = self.embedding_layer_norm(embedding_output)
        encoder_outputs = self.encoder(
            hidden_states,
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

class PreLnForMaskedLM(PreLnPreTrainedModel):
    config_class = PreLnForMaskedLMConfig
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `PreLnForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.fp32_cast_logits = config.fp32_cast_logits

        logger.info("PreLnForMaskedLM fp32_cast_layer_norm=%s", str(config.fp32_cast_layer_norm))
        logger.info("PreLnForMaskedLM fp32_cast_query_key=%s", str(config.fp32_cast_query_key))
        logger.info("PreLnForMaskedLM fp32_cast_attention_scores=%s", str(config.fp32_cast_attention_scores))
        logger.info("PreLnForMaskedLM fp32_cast_logits=%s", str(config.fp32_cast_logits))

        self.pre_ln_encoder = PreLnModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

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
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.pre_ln_encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            orig_dtype = prediction_scores.dtype
            if self.fp32_cast_logits:
                prediction_scores = prediction_scores.float()

            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

            # In case did cast.
            prediction_scores = prediction_scores.to(orig_dtype)
            masked_lm_loss = masked_lm_loss.to(orig_dtype)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
