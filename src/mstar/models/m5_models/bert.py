# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright 2018, NVIDIA CORPORATION.
# Copyright 2020, Amazon.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT Layers."""
 
import json
import enum
import copy
import os
import sys
import math
import torch
import torch.distributed
import torch.nn.functional as F
import torch.nn.init as init
 
from torch import nn
from torch.utils import checkpoint
from torch.nn import Module
from torch.nn.parameter import Parameter
from transformers.utils import logging
from torch.nn import LayerNorm as BertLayerNorm
from types import SimpleNamespace
from transformers.configuration_utils import PretrainedConfig
from mstar.megatron.fused_softmax import FusedScaleMaskSoftmax
 
logger = logging.get_logger(__name__)
 
try:
    from apex.normalization import FusedLayerNorm
 
    BertLayerNorm = FusedLayerNorm
 
    logger.info(
        "Discovered apex.normalization.FusedLayerNorm - will use it instead of BertLayerNorm"
    )
except ImportError:
    # using the normal BertLayerNorm
    pass
except Exception:
    logger.warning("discovered apex but it failed to load, falling back to BertLayerNorm")
    pass
 
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
ARGS_NAME = "deepspeed_config.json"
 
LABEL_IGNORE_INDEX = -100
@torch.jit.script
def f_gelu(x):
    pdtype = x.dtype
    x = x.float()
    y = x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
    return y.to(pdtype)
 
 
@torch.jit.script
def bias_gelu(bias, y):
    x = bias + y
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))
 
 
@torch.jit.script
def bias_tanh(bias, y):
    x = bias + y
    return torch.tanh(x)
 
 
@torch.jit.script
def h_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
 
 
def gelu(x):
    """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    Also see https://arxiv.org/abs/1606.08415
    """
    return h_gelu(x)
 
 
def swish(x):
    return x * torch.sigmoid(x)
 
 
ACT2FN = {
    "gelu": gelu,
    "relu": torch.nn.functional.relu,
    "tanh": torch.nn.functional.tanh,
    "swish": swish,
}
 
 
class AttnMaskType(enum.Enum):
    padding = 1
    causal = 2
 
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)
 
###### BIAS GELU FUSION/ NO AUTOGRAD ################
# 1/sqrt(2*pi)-> 0.3989423
# 1/sqrt(2)   -> 0.70710678
# sqrt(2/pi)  -> 0.79788456
# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))
 
@torch.jit.script
def bias_gelu(bias, y):
    x = bias + y
    return  x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))
 
# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.jit.script
def bias_gelu_back(g, bias, y):
    x = bias + y
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff*g
 
 
class GeLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_gelu(bias, input)
 
    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_gelu_back(grad_output, bias, input)
        return tmp, tmp
 
bias_gelu_impl = GeLUFunction.apply
 
 
class LinearActivation(nn.Module):
    r"""Fused Linear and activation Module."""
    __constants__ = ["bias"]
 
    def __init__(self, in_features, out_features, act="gelu", bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fused_gelu = False
        self.fused_tanh = False
        if isinstance(act, str) or (
            sys.version_info[0] == 2 and isinstance(act, unicode)
        ):
            if bias and act == "gelu":
                self.fused_gelu = True
            elif bias and act == "tanh":
                self.fused_tanh = True
            else:
                self.act_fn = ACT2FN[act]
        else:
            self.act_fn = act
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
 
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
 
    def forward(self, input):
        if self.fused_gelu:
            return bias_gelu_impl(F.linear(input, self.weight, None), self.bias)
            #return F.gelu(F.linear(input, self.weight, None) + self.bias)
        elif self.fused_tanh:
            return bias_tanh(self.bias, F.linear(input, self.weight, None))
        else:
            return self.act_fn(F.linear(input, self.weight, self.bias))
 
    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
 
class UnfusedBertLayerNorm(Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super().__init__()
        self.weight = Parameter(torch.ones(hidden_size))
        self.bias = Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
 
    def forward(self, x):
        pdtype = x.dtype
        x = x.float()
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x.to(pdtype) + self.bias
 
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
 
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        if config.type_vocab_size > 1:
            self.segment_embeddings = nn.Embedding(config.type_vocab_size,
                                                   config.hidden_size)
 
        if config.with_meta:
            self.LayerNorm = UnfusedBertLayerNorm(config.hidden_size, eps=1e-5)
        else:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
 
    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length,
                                    dtype=torch.long,
                                    device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
 
        words_embeddings = self.word_embeddings(input_ids)
        segment_embeddings = 0
        if token_type_ids is not None and hasattr(self, "segment_embeddings"):
            segment_embeddings = self.segment_embeddings(token_type_ids)
 
        position_embeddings = self.position_embeddings(position_ids)
 
        embeddings = words_embeddings + position_embeddings + segment_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
 
 
class BertSelfAttention(nn.Module):
    def __init__(self, config, layer_number=1):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
 
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
 
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)
 
        def enc_attention_mask_func(attention_scores, attention_mask):
            attention_scores.masked_fill_(attention_mask, -10000.0)
            return attention_scores
 
        if config.use_fused_softmax:
            self.scale_mask_softmax = FusedScaleMaskSoftmax(
                input_in_fp16=config.fp16,
                input_in_bf16=config.bf16,
                attn_mask_type=AttnMaskType.padding,
                scaled_masked_softmax_fusion=True,
                mask_func=enc_attention_mask_func,
                softmax_in_fp32=True,
                scale=None
            )
        self.scale_factor = 1. / (math.sqrt(self.attention_head_size))
        if config.scale_with_layer_number:
            self.scale_factor /= layer_number
 
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)
        x = x.view(*new_x_shape)
        return x
 
    def forward_back(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
 
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_key_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
 
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in M5BertModel forward() function)
        attention_scores = attention_scores + attention_mask
 
        pdtype = attention_scores.dtype
        # Normalize the attention scores to probabilities.
        attention_probs = self.softmax(attention_scores)
 
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
 
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
 
    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
 
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
 
        # Take the dot product between "query" and "key" to get the raw attention scores.
        # attention_scores = torch.matmul(query_layer, key_layer)
        # attention_scores = attention_scores / math.sqrt(
        #    self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in M5BertModel forward() function)
        # attention_scores = attention_scores + attention_mask
 
        # Normalize the attention scores to probabilities.
        # attention_probs = self.softmax(attention_scores)
 
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)
 
        ###########
 
        # [b, np, sq, sk]
        output_size = (query_layer.size(0),
                       query_layer.size(2),
                       query_layer.size(1),
                       key_layer.size(1))
 
        # [b, sq, np, hn] -> [b * np, sq, hn]
        query_layer = query_layer.transpose(1, 2).contiguous()
        query_layer = query_layer.view(output_size[0] * output_size[1],
                                       output_size[2], query_layer.size(3))
        # [b, sk, np, hn] -> [b * np, sk, hn]
        key_layer = key_layer.transpose(1, 2).contiguous()
        key_layer = key_layer.view(output_size[0] * output_size[1],
                                   key_layer.size(2), key_layer.size(3))
 
        # Raw attention scores. [b * np, sq, sk]
 
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device())
 
        matmul_result.baddbmm_(
            query_layer,  # [b * np, sq, hn]
            key_layer.transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0, alpha=self.scale_factor)
 
        attention_scores = matmul_result.view(*output_size)
 
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask.to(attention_scores.device))
 
 
        attention_probs = self.dropout(attention_probs)
 
        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(0),
                       value_layer.size(2),
                       query_layer.size(1),
                       value_layer.size(3))
 
        # change view [b * np, sk, hn]
        value_layer = value_layer.transpose(1, 2).contiguous()
        value_layer = value_layer.view(output_size[0] * output_size[1],
                                       value_layer.size(2), value_layer.size(3))
 
        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)
 
        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer)
 
        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
 
        # [b, np, sq, hn] --> [b, sq, np, hn]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        ###########
 
        # context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer
 
 
class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense.bert_output_layer = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
 
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
 
 
class BertAttention(nn.Module):
    def __init__(self, config, layer_number=1):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config, layer_number=layer_number)
        self.output = BertSelfOutput(config)
 
    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output
 
 
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense_act = LinearActivation(config.hidden_size,
                                          config.intermediate_size,
                                          act=config.hidden_act)
 
    def forward(self, hidden_states):
        hidden_states = self.dense_act(hidden_states)
        return hidden_states
 
 
class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dense.bert_output_layer = True
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
 
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
 
 
class BertLayer(nn.Module):
    def __init__(self, config, layer_number=1):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config, layer_number=layer_number)
        if config.with_meta:
            self.PreAttentionLayerNorm = UnfusedBertLayerNorm(config.hidden_size,
                                                              eps=1e-5)
            self.PostAttentionLayerNorm = UnfusedBertLayerNorm(config.hidden_size,
                                                               eps=1e-5)
        else:
            self.PreAttentionLayerNorm = BertLayerNorm(config.hidden_size,
                                                       eps=1e-5)
            self.PostAttentionLayerNorm = BertLayerNorm(config.hidden_size,
                                                        eps=1e-5)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
 
    def forward(self, hidden_states, attention_mask):
        input_layer_norm = self.PreAttentionLayerNorm(hidden_states)
        attention_output = self.attention(input_layer_norm, attention_mask)
 
        intermediate_input = hidden_states + attention_output
 
        intermediate_layer_norm = self.PostAttentionLayerNorm(
            intermediate_input)
 
        intermediate_output = self.intermediate(intermediate_layer_norm)
        layer_output = self.output(intermediate_output)
 
        return layer_output + intermediate_input
 
 
class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense_act = LinearActivation(config.hidden_size,
                                          config.hidden_size,
                                          act="tanh")
 
    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense_act(first_token_tensor)
        return pooled_output
 
 
class BertEncoder(nn.Module):
    def __init__(self, config, args, sparse_attention_config=None):
        super().__init__()
        # Added later to make it similar to GPT-2
        if config.with_meta:
            self.FinalLayerNorm = UnfusedBertLayerNorm(config.hidden_size, eps=1e-5)
        else:
            self.FinalLayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
 
        layers = []
        for layer_number in range(config.num_hidden_layers):
            layer = BertLayer(config, layer_number=layer_number + 1)
            if sparse_attention_config is not None:
                from deepspeed.ops.sparse_attention import BertSparseSelfAttention
 
                layer.attention.self = BertSparseSelfAttention(
                    config, sparsity_config=sparse_attention_config
                )
            layers.append(layer)
        self.layer = nn.ModuleList(layers)
 
    def forward(
        self,
        hidden_states,
        attention_mask,
        output_all_encoded_layers=True,
        checkpoint_activations=False,
    ):
        all_encoder_layers = []
 
        def custom(start, end):
            def custom_forward(*inputs):
                layers = self.layer[start:end]
                x_ = inputs[0]
                for layer in layers:
                    x_ = layer(x_, inputs[1])
                return x_
 
            return custom_forward
 
        if checkpoint_activations:
            l = 1
            num_layers = len(self.layer)
            #chunk_length = math.ceil(math.sqrt(num_layers))
            chunk_length = 1
            while l < num_layers:
                hidden_states = checkpoint.checkpoint(
                    custom(l, l + chunk_length), hidden_states, attention_mask
                )
                l += chunk_length
            # decoder layers
        else:
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(hidden_states, attention_mask)
 
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
 
        if not output_all_encoded_layers or checkpoint_activations:
            hidden_states = self.FinalLayerNorm(hidden_states)
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
 
 
class BertPreTrainedModel(nn.Module):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """
 
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, M5BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `M5BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        self.config = config

    def init_bert_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding, LinearActivation)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            num_layers = self.config.num_hidden_layers
            std = self.config.initializer_range
            if hasattr(module, "bert_output_layer"):
                # "Accounting for accumulation on the residual path"
                # print("Accounting for accumulation on the residual path")
                std = self.config.initializer_range / math.sqrt(2.0 * num_layers)
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, (BertLayerNorm, UnfusedBertLayerNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, (nn.Linear, LinearActivation)) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        state_dict=None,
        cache_dir=None,
        from_tf=False,
        *inputs,
        **kwargs,
    ):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Arguments:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `model_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `model_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ", ".join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file,
                )
            )
            return None
        if resolved_archive_file == archive_file:
            print("loading archive file {}".format(archive_file))
        else:
            print(
                "loading archive file {} from cache at {}".format(
                    archive_file, resolved_archive_file
                )
            )
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            print(
                "extracting archive file {} to temp dir {}".format(
                    resolved_archive_file, tempdir
                )
            )
            with tarfile.open(resolved_archive_file, "r:gz") as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = M5BertConfig.from_json_file(config_file)
        print("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(
                weights_path,
                map_location="cpu" if not torch.cuda.is_available() else None,
            )
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace("gamma", "weight")
            if "beta" in key:
                new_key = key.replace("beta", "bias")
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        start_prefix = ""
        if not hasattr(model, "bert") and any(
            s.startswith("bert.") for s in state_dict.keys()
        ):
            start_prefix = "bert."
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            print(
                "Weights of {} not initialized from pretrained model: {}".format(
                    model.__class__.__name__, missing_keys
                )
            )
        if len(unexpected_keys) > 0:
            print(
                "Weights from pretrained model not used in {}: {}".format(
                    model.__class__.__name__, unexpected_keys
                )
            )
        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        return model


def gen_self_attn_mask(data, valid_length=None, dtype=None, attn_type: str = "full", additional_length=None):
    """Generate the mask used for the encoder, i.e, self-attention.

    In our implementation, 1 --> not masked, 0 --> masked

    Let's consider the data with two samples:

    data =
        [['My', 'dog', 'is ', 'cute'],
         ['I', 'agree', '[PAD]', '[PAD]']]
    valid_length =
        [4, 2]

    - attn_type = 'causal'
        Each token will attend to itself + the tokens before.
        It will not attend to tokens in the future.

        The mask of the first sample is
                   'My', 'dog', 'is ', 'cute'
        'My':        1,    0,     0,     0
        'dog':       1,    1,     0,     0
        'is':        1,    1,     1,     0
        'cute':      1,    1,     1,     1

        The mask of the second sample is
                   'I', 'agree', '[PAD]', '[PAD]'
        'I':        1,     0,       0,       0
        'agree':    1,     1,       0,       0
        '<PAD>':    0,     0,       0,       0
        '<PAD>':    0,     0,       0,       0


    - attn_type = 'full'
        Each token will attend to both the tokens before and in the future

        The mask of the first sample is
                   'My', 'dog', 'is ', 'cute'
        'My':        1,    1,     1,     1
        'dog':       1,    1,     1,     1
        'is':        1,    1,     1,     1
        'cute':      1,    1,     1,     1

        The mask of the second sample is
                   'I', 'agree', '[PAD]', '[PAD]'
        'I':        1,     1,       0,       0
        'agree':    1,     1,       0,       0
        '<PAD>':    0,     0,       0,       0
        '<PAD>':    0,     0,       0,       0

    Arguments:
        data (torch.Tensor): Shape (batch_size, seq_length, ...)
        valid_length (torch.Tensor, optional): Shape (batch_size,). (default: ``None`` )
        dtype (torch.dtype, optional): torch data type of the mask. (default: ``None`` )
        attn_type (str, optional): Can be 'full' or 'causal'. (default: ``'full'`` )
        additional_length (int, optional): optional additional length. (default: ``None`` )

    Returns:
        if attn_type = "full", mask: Shape (batch_size, seq_length, seq_length), where
        1 means a singleton dimension for broadcasting.
        Otherwise, mask: Shape (batch_size, seq_length, seq_length)
    """
    length = data.size(1)
    dtype = data.dtype if dtype is None else dtype
    if additional_length is None:
        additional_length = 0
 
    if attn_type == "full":
        if valid_length is not None:
            steps = torch.arange(0, length + additional_length, device=data.device)
            mask = torch.logical_or(steps.reshape((1, 1, -1)) < valid_length.reshape((-1, 1, 1)),
                                    steps.reshape((1, 1, -1)) >= torch.empty((valid_length.size(0), 1, 1), device=data.device).fill_(length))
            mask = mask.expand((mask.shape[0], mask.shape[2], mask.shape[2]))
            mask = mask * mask.transpose(1, 2)
        else:
            mask = torch.ones((data.size(0),
                               length + additional_length,
                               length + additional_length),
                              dtype=dtype)
    elif attn_type == "causal":
        steps = torch.arange(0, length + additional_length, device=data.device)
        # mask: (seq_length, seq_length)
        # batch_mask: (batch_size, seq_length)
        mask = torch.logical_or(steps.unsqueeze(0) <= steps.unsqueeze(1),
                                steps.unsqueeze(0) >= torch.empty((steps.size(0), 1), device=data.device).fill_(length))
        mask = mask.to(dtype=dtype)
        if valid_length is not None:
            batch_mask = torch.logical_or(steps.unsqueeze(0) < valid_length.unsqueeze(1),
                                          steps.unsqueeze(0) >= torch.empty((valid_length.size(0), 1), device=data.device).fill_(length))
            batch_mask = batch_mask.to(dtype=dtype)
            mask = mask * batch_mask.unsqueeze(-1)
        else:
            batch_ones = torch.ones(
                data.size(0), dtype=dtype, device=mask.device
            )  # (batch_size,)
            mask = mask * batch_ones.reshape((-1, 1, 1))
    else:
        raise NotImplementedError

    return mask.to(dtype=dtype)


class M5BertConfig(PretrainedConfig):
    """Configuration class to store the configuration of a `M5BertModel`.

    Args:
        vocab_size: Vocabulary size of `inputs_ids` in `M5BertModel`.
        hidden_size: Size of the encoder layers and the pooler layer.
        num_hidden_layers: Number of hidden layers in the Transformer encoder.
        num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
        intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
        hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
        hidden_dropout_prob: The dropout probabilitiy for all fully connected
            layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
        max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
        initializer_range: The sttdev of the truncated_normal_initializer for
            initializing all weight matrices.
        type_vocab_size: The maximum type of tokens used in bert. Default = 1.
        checkpoint_activations: Choose to load from an existing checkpoint.
        fp16: if fp16 is used for the training.  When use_fused_softmax is True, 
            it is recommended to use either fp16 or bf16.
        bf16: if bf16 is used for the training.
        use_fused_softmax: if fused softmax kernel is enabled. This kernels
            requires that the sequence length is a multiple of 4.
        output_transform: a string to choose the transform_function to change
            output of bert before returning it. Default = None. Options include
            any transformation [pool, avg, seq]. If set, input value for
            `output_all_encoded_layers` will be ignored.
        scale_with_layer_number: if attention will be scaled by the inverse of
            layer idx. This makes the training of large models more stable. 
            Reference (Megatron-LM): 
            https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
    """
    model_type = "mstar-bert"

    def __init__(
        self,
        vocab_size=512035,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        type_vocab_size=1,
        with_meta=False,
        checkpoint_activations=False,
        fp16=False,
        bf16=False,
        use_fused_softmax=False,
        output_transform=None,
        scale_with_layer_number=False,
        **kwargs
    ):
        super().__init__()

        self.with_meta = with_meta
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.checkpoint_activations = checkpoint_activations
        self.fp16 = fp16
        self.bf16 = bf16
        self.use_fused_softmax = use_fused_softmax
        self.output_transform = output_transform
        self.scale_with_layer_number = scale_with_layer_number
        if self.fp16 and self.bf16:
            raise ValueError("both fp16 and bf16 flags cannot be active at the same time.")


class M5BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Arguments:
        config: a M5BertConfig class instance with the configuration to build a new model

    Inputs:
        input_ids (torch.LongTensor): Shape [batch_size, sequence_length]
            with the word token indices in the vocabulary.
        valid_length (torch.LongTensor, optional): Shape [batch_size, ]
            with sequence lengths. It's used to create self-attention mask.
            It's typically used for attention when a batch has varying length sentences.
        token_type_ids (torch.LongTensor, optional): Shape [batch_size, sequence_length]
            with token id value less than config.type_vocab_size.
            It's typically used to distinguish different components in input.
        output_all_encoded_layers (boolean, optional): controls the content of the
            `encoded_layers` output as described below. (default: ``False``)
        checkpoint_activations (boolean): whether to use gradient checkpointing for reducing
            memory footprint. (default: ``False``)

    Outputs: Tuple of (encoded_layers, pooled_output)
        encoded_layers: controlled by :attr:`output_all_encoded_layers` argument:
            - output_all_encoded_layers=``True``:
                outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - output_all_encoded_layers=``False``: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        pooled_output (torch.FloatTensor): Shape [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (``[CLS]``).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    >>> from m5_transformers import M5BertConfig, M5BertModel
    >>> input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    >>> valid_length = torch.LongTensor([3, 2])
    >>>
    >>> config = M5BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
    >>>                     num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    >>>
    >>> model = M5BertModel(config=config)
    >>> all_encoder_layers, pooled_output = model(input_ids, valid_length)
    ```
    """

    config_class = M5BertConfig

    def __init__(self, config, ds_args):
        super().__init__(config, ds_args)
        self.embeddings = BertEmbeddings(config)
        # set pad_token_id that is used for sparse attention padding
        self.pad_token_id = (
            config.pad_token_id
            if hasattr(config, "pad_token_id") and config.pad_token_id is not None
            else 0
        )
        # set sparse_attention_config if it has been selected
        self.sparse_attention_config = None
        self.sparse_attention_utils = None
        self.encoder = BertEncoder(
            config, ds_args, sparse_attention_config=self.sparse_attention_config
        )
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(
        self,
        input_ids,
        valid_length=None,
        token_type_ids=None,
        additional_inputs=None,
        output_all_encoded_layers=False,
        checkpoint_activations=False,
    ):
        # We create a 4D attention mask from a 3D tensor mask output by gen_self_attn_mask.
        # Sizes are [batch_size, 1, from_seq_length, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        attention_mask = gen_self_attn_mask(
            input_ids, valid_length=valid_length, dtype=next(self.parameters()).dtype,
            additional_length=additional_inputs.size(1) if additional_inputs is not None else None
        )  # fp16 compatibility
        extended_attention_mask = attention_mask.unsqueeze(1)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and 1 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        #extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        extended_attention_mask = (1.0 - extended_attention_mask)
        extended_attention_mask = extended_attention_mask.type(torch.bool)

        # If BertEncoder uses sparse attention, it needs to be padded based on the sparse attention block size
        if self.sparse_attention_config is not None:
            token_type_ids = (
                torch.zeros_like(input_ids) if not token_type_ids else token_type_ids
            )
            (
                pad_len,
                input_ids,
                attention_mask,
                _,
                position_ids,
                inputs_embeds,
            ) = self.sparse_attention_utils.pad_to_block_size(
                block_size=self.sparse_attention_config.block,
                input_ids=input_ids,
                attention_mask=extended_attention_mask,
                token_type_ids=token_type_ids,
                position_ids=None,
                inputs_embeds=None,
                pad_token_id=self.pad_token_id,
                model_mbeddings=self.embeddings,
            )

        embedding_output = self.embeddings(input_ids, token_type_ids)
        if additional_inputs is not None:
            embedding_output = torch.cat((embedding_output, additional_inputs), 1)
        encoded_layers = self.encoder(
            embedding_output,
            extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers,
            checkpoint_activations=checkpoint_activations,
        )
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)

        # If BertEncoder uses sparse attention, and input_ids were padded, sequence output needs to be unpadded to original length
        if self.sparse_attention_config is not None and pad_len > 0:
            encoded_layers[-1] = self.sparse_attention_utils.unpad_sequence_output(
                pad_len, encoded_layers[-1]
            )

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


def gen_sequence_mask(max_seq_len, valid_len, dtype=torch.int):
    """
    Given max_seq_length and a 1-D tensor with valid lengths, this function will
    return a binary 2-D tensor with ones for when index is less than valid_length.
    Example:
        max_seq_len = 4
        valid_len = torch.tensor([1, 3, 2])
        return: mask = torch.tensor([[1, 0, 0, 0],
                                     [1, 1, 1, 0],
                                     [1, 1, 0, 0]])
    Input:
        max_seq_len: Maximum sequence length possible, valid_len cannot have a number
            larger than this.
        valid_len: A 1-D tensor with integer in range 0, max_seq_len - 1
        dtype: The tensor type to be returned. Default = torch.int.
    Output:
        Return a 2-D tensor of shape (valid_len.size(0), max_seq_len)
    """
    if len(valid_len.size()) != 1:
        raise ValueError(
            f"valid_len must be a 1-d tensor. Given shape: {valid_len.size()}"
        )
    batch_size = valid_len.size(0)
    mask = torch.arange(0, max_seq_len, device=valid_len.device)
    mask = mask.repeat(batch_size).view((-1, max_seq_len))
    mask = (mask < valid_len.unsqueeze(1)).to(dtype)
    return mask

MIN_DIVISION_VALUE = 1e-9
def mean_pooling(sequenced_output, valid_len=None, attention_mask=None):
    """
    Compute mean pooling given tensor shape (batch_size, max_seq_len, hidden_dim).
    Function adapted from https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens
    Input:
        sequenced_output: An n-D tensor with n >= 3. Tensor roughly of shape
            (batch_size, max_seq_len, hidden_dim, *)
        valid_len: A 1-D tensor with integer representing valid length for each
            row in sequenced_output. valid_len.size(0) = batch_size. If attention_mask
            is set valid_len cannot be passed.
        attention_mask: A 2-D tensor with ones and zeros. If valid_len is set
            attention_mask cannot be passed.
    Output:
        Returns a mean embedding along the 1st dimension. Tensor returned of shape
            (batch_size, hidden_dim)
    """
    if len(sequenced_output.size()) != 3:
        raise ValueError(
            f"sequenced_output must be a 3-D tensor. "
            f"Given shape: {sequenced_output.size()}"
        )

    if attention_mask is not None and valid_len is not None:
        raise ValueError("Cannot pass both attention_mask and valid_length.")

    if attention_mask is None and valid_len is None:
        return sequenced_output.mean(1)

    max_seq_len = sequenced_output.size(1)
    if valid_len is not None:
        attention_mask = gen_sequence_mask(max_seq_len, valid_len)
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(sequenced_output.size()).float()
    )
    sum_embeddings = torch.sum(sequenced_output * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=MIN_DIVISION_VALUE)
    return (sum_embeddings / sum_mask).type(sequenced_output.dtype)


def mean_seq_output(sequenced_output, pooled_output, valid_len):
    """
    The function helps transform bert output for models that use bert output from one entity.
    Returns average of sequenced_output, expect sequence tensor of shape (batch, seq_len, hidden_dim).
    """
    return mean_pooling(sequenced_output, valid_len=valid_len)


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense_act = LinearActivation(
            config.hidden_size, config.hidden_size, act=config.hidden_act
        )
        if config.with_meta:
            self.LayerNorm = UnfusedBertLayerNorm(config.hidden_size, eps=1e-5)
        else:
            self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, hidden_states):
        hidden_states = self.dense_act(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        #self.decoder = nn.Linear(
        #    bert_model_embedding_weights.size(1),
        #    bert_model_embedding_weights.size(0),
        #    bias=False,
        #)
        self.decoder = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        self.decoder.weight = bert_model_embedding_weights
        #self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states, masked_token_indexes):
        hidden_states = self.transform(hidden_states)

        if masked_token_indexes is not None:
            hidden_states = torch.index_select(
                hidden_states.view(-1, hidden_states.shape[-1]), 0, masked_token_indexes
            )

        torch.cuda.nvtx.range_push(
            "decoder input.size() = {}, weight.size() = {}".format(
                hidden_states.size(), self.decoder.weight.size()
            )
        )
        hidden_states = self.decoder(hidden_states) + self.bias
        torch.cuda.nvtx.range_pop()
        return hidden_states


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output, pooled_output, masked_token_indexes=None):
        prediction_scores = self.predictions(sequence_output, masked_token_indexes)
        return prediction_scores


class M5BertForPreTrainingPreLN(BertPreTrainedModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by one pre-training head:
        - the masked language modeling head

    Arguments:
        config: a M5BertConfig class instance with the configuration to build a new model.
        args: user arguments mainly to check model parallel, sparse attention, and deepspeed config.

    Inputs:
        input_ids (torch.LongTensor): Shape [batch_size, sequence_length]
            with the word token indices in the vocabulary.
        valid_length (torch.LongTensor, optional): Shape [batch_size, ]
            with sequence lengths. It's used to create self-attention mask.
            It's typically used for attention when a batch has varying length sentences.
        masked_lm_labels (torch.LongTensor, optional): masked language modeling labels of
            shape [batch_size, sequence_length] with indices selected
            in [-1, 0, ..., vocab_size - 1]. All labels set to -1 are ignored (masked),
            the loss is only computed for the labels set in [0, ..., vocab_size - 1]

    Outputs:
        if masked_lm_labels is provided and embedding_mode is False:
            Outputs the masked language modeling loss.
        if embedding_mode is ``True``:
            Outputs a tuple comprising
            - tokens embeddings and output of pooler.
        if masked_lm_labels is ``None`` and model_parallel is False:
            Outputs the masked language modeling logits of shape
            [batch_size, sequence_length, vocab_size]
        if masked_lm_labels is ``None`` and model_parallel is True:
            Outputs the masked language modeling logits of shape
            [batch_size, sequence_length, vocab_size // model_parallel_size]

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    >>> from m5_transformers import M5BertConfig, BertForPreTraining
    >>> input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    >>> valid_length = torch.LongTensor([3, 2])
    >>>
    >>> config = M5BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
    >>>                     num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    >>>
    >>> model = BertForPreTraining(config)
    >>> masked_lm_logits_scores = model(input_ids, valid_length)
    ```
    """

    config_class = M5BertConfig

    def __init__(self, config, ds_args=None):
        super().__init__(config, ds_args)
        self.bert = M5BertModel(config, ds_args)
        self.embedding_dim, self.output_transform = config.hidden_size, mean_seq_output
        try:
            import deepspeed
            deepspeed.zero.register_external_parameter(self, self.bert.embeddings.word_embeddings.weight)
        except ImportError:
            pass
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight
        )
        self.apply(self.init_bert_weights)
        self.args = ds_args

    def forward(self, input_ids,
                valid_length=None,
                labels=None,
                embedding_mode=False,
                checkpoint_activations=False):

        sequence_output, pooled_output = self.bert(input_ids,
                                                   valid_length,
                                                   checkpoint_activations=checkpoint_activations)

        if embedding_mode:
            return self.output_transform(sequence_output, pooled_output, valid_length)

        if labels is not None:
            # filter out all masked labels.
            masked_token_indexes = torch.nonzero(
                (labels - LABEL_IGNORE_INDEX).view(-1)
            ).view(-1)
            prediction_scores = self.cls(
                sequence_output, pooled_output, masked_token_indexes
            )
            target = torch.index_select(labels.view(-1), 0, masked_token_indexes)

            loss_fct = nn.CrossEntropyLoss(ignore_index=LABEL_IGNORE_INDEX)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), target
            )

            total_loss = masked_lm_loss
            return total_loss
        else:
            masked_lm_logits_scores = self.cls(sequence_output, pooled_output)
            return masked_lm_logits_scores

    @staticmethod
    def from_pretrained(folder_loc, config=None, args=None, **kwargs):
        config_file_loc = os.path.join(folder_loc, CONFIG_NAME)
        model_file_loc = os.path.join(folder_loc, WEIGHTS_NAME)
        args_file_loc = os.path.join(folder_loc, ARGS_NAME)
        model = torch.load(model_file_loc)
        if not config:
            config = M5BertConfig.from_json_file(config_file_loc)

        class miniArgs:
            def __init__(self):
                self.deepspeed_transformer_kernel = False
                self.deepspeed_sparse_attention = False
                self.zero = 2

        if args is None and not os.path.exists(args_file_loc):
            print("Assuming only model initialized without deepspeed kernel.")
            args = miniArgs()
        elif os.path.exists(args_file_loc):
            with open(args_file_loc) as f:
                args = json.load(f, object_hook=lambda d: SimpleNamespace(**d))
        pt_model = M5BertForPreTrainingPreLN(config, args)
        pt_model.load_state_dict(model)
        return pt_model

    def save_pretrained(self, folder_loc):
        os.makedirs(folder_loc, exist_ok=True)
        config_file_loc = os.path.join(folder_loc, CONFIG_NAME)
        model_file_loc = os.path.join(folder_loc, WEIGHTS_NAME)
        args_file_loc = os.path.join(folder_loc, ARGS_NAME)
        with open(config_file_loc, "w") as of:
            json.dump(self.config.to_dict(), of)
        # with open(args_file_loc, "w") as of:
        #     json.dump(self.args.__dict__, of)
        torch.save(self.state_dict(), model_file_loc)
