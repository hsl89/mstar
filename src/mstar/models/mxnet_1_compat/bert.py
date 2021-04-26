"""BERT models."""

__all__ = ['BERTEncoder', 'BERTModel']

import mxnet as mx

from mxnet.gluon import HybridBlock, nn

###############################################################################
#                              COMPONENTS                                     #
###############################################################################


class PositionwiseFFN(HybridBlock):
    """Positionwise Feed-Forward Neural Network.

    Parameters
    ----------
    units : int
        Number of units for the output
    hidden_size : int
        Number of units in the hidden layer of position-wise feed-forward networks
    dropout : float
        Dropout probability for the output
    use_residual : bool
        Add residual connection between the input and the output
    ffn1_dropout : bool, default False
        If True, apply dropout both after the first and second Positionwise
        Feed-Forward Neural Network layers. If False, only apply dropout after
        the second.
    activation : str, default 'relu'
        Activation function
    layer_norm_eps : float, default 1e-5
        Epsilon parameter passed to for mxnet.gluon.nn.LayerNorm
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default None
        Prefix for name of `Block`s
        (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells.
        Created if `None`.
    """
    def __init__(self, *, units=512, hidden_size=2048, dropout=0.0, use_residual=True,
                 ffn1_dropout=False, activation='relu', layer_norm_eps=1e-5,
                 weight_initializer=None, bias_initializer='zeros', pre_norm: bool = False,
                 prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self._use_residual = use_residual
        self._dropout = dropout
        self._pre_norm = pre_norm
        self._ffn1_dropout = ffn1_dropout
        with self.name_scope():
            self.ffn_1 = nn.Dense(units=hidden_size, flatten=False,
                                  weight_initializer=weight_initializer,
                                  bias_initializer=bias_initializer, prefix='ffn_1_')
            assert activation == 'gelu'
            self.ffn_2 = nn.Dense(units=units, flatten=False, weight_initializer=weight_initializer,
                                  bias_initializer=bias_initializer, prefix='ffn_2_')
            if dropout:
                self.dropout_layer = nn.Dropout(rate=dropout)
            self.layer_norm = nn.LayerNorm(in_channels=units, epsilon=layer_norm_eps)

    def hybrid_forward(self, F, inputs):  # pylint: disable=arguments-differ
        """Position-wise encoding of the inputs.

        Parameters
        ----------
        inputs : Symbol or NDArray
            Input sequence. Shape (batch_size, length, C_in)

        Returns
        -------
        outputs : Symbol or NDArray
            Shape (batch_size, length, C_out)
        """
        if self._pre_norm:
            outputs = self.layer_norm(inputs)
        else:
            outputs = inputs
        outputs = self.ffn_1(outputs)
        outputs = F.LeakyReLU(outputs, act_type='gelu')
        if self._dropout and self._ffn1_dropout:
            outputs = self.dropout_layer(outputs)
        outputs = self.ffn_2(outputs)
        if self._dropout:
            outputs = self.dropout_layer(outputs)
        if self._use_residual:
            outputs = outputs + inputs
        if not self._pre_norm:
            outputs = self.layer_norm(outputs)
        return outputs


class DotProductSelfAttentionCell(HybridBlock):
    r"""Multi-head Dot Product Self Attention Cell.

    In the DotProductSelfAttentionCell, the input query/key/value will be linearly projected
    for `num_heads` times with different projection matrices. Each projected key, value, query
    will be used to calculate the attention weights and values. The output of each head will be
    concatenated to form the final output.

    This is a more efficient implementation of MultiHeadAttentionCell with
    DotProductAttentionCell as the base_cell:

    score = <W_q h_q, W_k h_k> / sqrt(dim_q)

    Parameters
    ----------
    units : int
        Total number of projected units for query. Must be divided exactly by num_heads.
    num_heads : int
        Number of parallel attention heads
    use_bias : bool, default True
        Whether to use bias when projecting the query/key/values
    weight_initializer : str or `Initializer` or None, default None
        Initializer of the weights.
    bias_initializer : str or `Initializer`, default 'zeros'
        Initializer of the bias.
    prefix : str or None, default None
        See document of `Block`.
    params : str or None, default None
        See document of `Block`.

    Inputs:
      - **qkv** : Symbol or NDArray
        Query / Key / Value vector. Shape (query_length, batch_size, C_in)
      - **valid_len** : Symbol or NDArray or None, default None
        Valid length of the query/key/value slots. Shape (batch_size, query_length)

    Outputs:
      - **context_vec** : Symbol or NDArray
        Shape (query_length, batch_size, context_vec_dim)
      - **att_weights** : Symbol or NDArray
        Attention weights of multiple heads.
        Shape (batch_size, num_heads, query_length, memory_length)
    """
    def __init__(self, units, num_heads, dropout=0.0, use_bias=True, weight_initializer=None,
                 bias_initializer='zeros', prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self._num_heads = num_heads
        self._use_bias = use_bias
        self._dropout = dropout
        self.units = units
        with self.name_scope():
            if self._use_bias:
                self.query_bias = self.params.get('query_bias', shape=(self.units, ),
                                                  init=bias_initializer)
                self.key_bias = self.params.get('key_bias', shape=(self.units, ),
                                                init=bias_initializer)
                self.value_bias = self.params.get('value_bias', shape=(self.units, ),
                                                  init=bias_initializer)
            weight_shape = (self.units, self.units)
            self.query_weight = self.params.get('query_weight', shape=weight_shape,
                                                init=weight_initializer, allow_deferred_init=True)
            self.key_weight = self.params.get('key_weight', shape=weight_shape,
                                              init=weight_initializer, allow_deferred_init=True)
            self.value_weight = self.params.get('value_weight', shape=weight_shape,
                                                init=weight_initializer, allow_deferred_init=True)
            self.dropout_layer = nn.Dropout(self._dropout)

    def _collect_params_with_prefix(self, prefix=''):
        # the registered parameter names in v0.8 are the following:
        # prefix_proj_query.weight, prefix_proj_query.bias
        # prefix_proj_value.weight, prefix_proj_value.bias
        # prefix_proj_key.weight, prefix_proj_key.bias
        # this is a temporary fix to keep backward compatibility, due to an issue in MXNet:
        # https://github.com/apache/incubator-mxnet/issues/17220
        if prefix:
            prefix += '.'
        ret = {prefix + 'proj_' + k.replace('_', '.'): v for k, v in self._reg_params.items()}
        for name, child in self._children.items():
            ret.update(child._collect_params_with_prefix(prefix + name))
        return ret

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, qkv, valid_len, query_bias, key_bias, value_bias, query_weight,
                       key_weight, value_weight):
        # interleaved_matmul_selfatt ops assume the projection is done with interleaving
        # weights for query/key/value. The concatenated weight should have shape
        # (num_heads, C_out/num_heads * 3, C_in).
        query_weight = query_weight.reshape(shape=(self._num_heads, -1, 0), reverse=True)
        key_weight = key_weight.reshape(shape=(self._num_heads, -1, 0), reverse=True)
        value_weight = value_weight.reshape(shape=(self._num_heads, -1, 0), reverse=True)
        in_weight = F.concat(query_weight, key_weight, value_weight, dim=-2)
        in_weight = in_weight.reshape(shape=(-1, 0), reverse=True)
        # concat bias
        query_bias = query_bias.reshape(shape=(self._num_heads, -1), reverse=True)
        key_bias = key_bias.reshape(shape=(self._num_heads, -1), reverse=True)
        value_bias = value_bias.reshape(shape=(self._num_heads, -1), reverse=True)
        in_bias = F.stack(query_bias, key_bias, value_bias, axis=1).reshape(-1)

        # qkv_proj shape = (seq_length, batch_size, num_heads * head_dim * 3)
        qkv_proj = F.FullyConnected(data=qkv, weight=in_weight, bias=in_bias,
                                    num_hidden=self.units * 3, no_bias=False, flatten=False)
        att_score = F.contrib.interleaved_matmul_selfatt_qk(qkv_proj, heads=self._num_heads)
        if valid_len is not None:
            valid_len = F.broadcast_axis(F.expand_dims(valid_len, axis=1), axis=1,
                                         size=self._num_heads)
            valid_len = valid_len.reshape(shape=(-1, 0), reverse=True)
            att_weights = F.softmax(att_score, length=valid_len, use_length=True, axis=-1)
        else:
            att_weights = F.softmax(att_score, axis=-1)
        # att_weights shape = (batch_size, seq_length, seq_length)
        att_weights = self.dropout_layer(att_weights)
        context_vec = F.contrib.interleaved_matmul_selfatt_valatt(qkv_proj, att_weights,
                                                                  heads=self._num_heads)
        att_weights = att_weights.reshape(shape=(-1, self._num_heads, 0, 0), reverse=True)
        return context_vec, att_weights


class BERTEncoderCell(HybridBlock):
    """Structure of the BERT Encoder Cell.

    Parameters
    ----------
    units : int
        Number of units for the output
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    num_heads : int
        Number of heads in multi-head attention
    dropout : float
    output_attention: bool
        Whether to output the attention weights
    attention_use_bias : float, default True
        Whether to use bias term in the attention cell
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default None
        Prefix for name of `Block`s. (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells. Created if `None`.
    activation : str, default 'gelu'
        Activation methods in PositionwiseFFN
    layer_norm_eps : float, default 1e-5
        Epsilon for layer_norm

    Inputs:
        - **inputs** : input sequence. Shape (length, batch_size, C_in)
        - **valid_length** : valid length of inputs for attention. Shape (batch_size, length)

    Outputs:
        - **outputs**: output tensor of the transformer encoder cell.
            Shape (length, batch_size, C_out)
        - **additional_outputs**: the additional output of all the BERT encoder cell.
    """
    def __init__(self, units=128, hidden_size=512, num_heads=4, dropout=0.0, output_attention=False,
                 attention_use_bias=True, weight_initializer=None, bias_initializer='zeros',
                 prefix=None, params=None, activation='gelu', layer_norm_eps=1e-5, pre_norm=False):
        super().__init__(prefix=prefix, params=params)
        self._dropout = dropout
        self._output_attention = output_attention
        self._pre_norm = pre_norm
        with self.name_scope():
            if dropout:
                self.dropout_layer = nn.Dropout(rate=dropout)
            self.attention_cell = DotProductSelfAttentionCell(units, num_heads,
                                                              use_bias=attention_use_bias,
                                                              dropout=dropout)
            self.proj = nn.Dense(units=units, flatten=False, use_bias=True,
                                 weight_initializer=weight_initializer,
                                 bias_initializer=bias_initializer, prefix='proj_')
            self.ffn = PositionwiseFFN(units=units, hidden_size=hidden_size, dropout=dropout,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer, activation=activation,
                                       layer_norm_eps=layer_norm_eps, pre_norm=pre_norm)
            self.layer_norm = nn.LayerNorm(in_channels=units, epsilon=layer_norm_eps)

    def hybrid_forward(self, F, inputs, valid_len=None):  # pylint: disable=arguments-differ
        """Transformer Encoder Attention Cell.

        Parameters
        ----------
        inputs : Symbol or NDArray
            Input sequence. Shape (length, batch_size, C_in)
        valid_len : Symbol or NDArray or None
            Valid length for inputs. Shape (batch_size, length)

        Returns
        -------
        encoder_cell_outputs: list
            Outputs of the encoder cell. Contains:

            - outputs of the transformer encoder cell. Shape (length, batch_size, C_out)
            - additional_outputs of all the transformer encoder cell
        """
        if self._pre_norm:
            outputs = self.layer_norm(inputs)
        else:
            outputs = inputs
        outputs, attention_weights = self.attention_cell(outputs, valid_len)
        outputs = self.proj(outputs)
        if self._dropout:
            outputs = self.dropout_layer(outputs)
        # use residual
        outputs = outputs + inputs
        if not self._pre_norm:
            outputs = self.layer_norm(outputs)
        outputs = self.ffn(outputs)
        additional_outputs = []
        if self._output_attention:
            additional_outputs.append(attention_weights)
        return outputs, additional_outputs


class BERTEncoder(HybridBlock):
    """Structure of the BERT Encoder.

    Different from the original encoder for transformer, `BERTEncoder` uses
    learnable positional embedding, a 'gelu' activation functions and a
    separate epsilon value for LayerNorm.

    Parameters
    ----------
    num_layers : int
        Number of attention layers.
    units : int
        Number of units for the output.
    hidden_size : int
        number of units in the hidden layer of position-wise feed-forward networks
    max_length : int
        Maximum length of the input sequence
    num_heads : int
        Number of heads in multi-head attention
    dropout : float
        Dropout probability of the attention probabilities and embedding.
    output_attention: bool, default False
        Whether to output the attention weights
    output_all_encodings: bool, default False
        Whether to output encodings of all encoder cells
    weight_initializer : str or Initializer
        Initializer for the input weights matrix, used for the linear
        transformation of the inputs.
    bias_initializer : str or Initializer
        Initializer for the bias vector.
    prefix : str, default None.
        Prefix for name of `Block`s. (and name of weight if params is `None`).
    params : Parameter or None
        Container for weight sharing between cells. Created if `None`.
    activation : str, default 'gelu'
        Activation methods in PositionwiseFFN
    layer_norm_eps : float, default 1e-12
        Epsilon for layer_norm

    Inputs:
        - **inputs** : input sequence of shape (length, batch_size, C_in)
        - **states** : list of tensors for initial states and valid length for self attention.
        - **valid_length** : valid lengths of each sequence. Usually used when part of sequence
            has been padded. Shape is (batch_size, )

    Outputs:
        - **outputs** : the output of the encoder. Shape is (length, batch_size, C_out)
        - **additional_outputs** : list of tensors.
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, num_heads, length, mem_length)

    """
    def __init__(self, *, num_layers=2, units=512, hidden_size=2048, max_length=50, num_heads=4,
                 dropout=0.0, output_attention=False, output_all_encodings=False,
                 weight_initializer=None, bias_initializer='zeros', prefix=None, params=None,
                 activation='gelu', layer_norm_eps=1e-12, pre_norm=False):
        super().__init__(prefix=prefix, params=params)
        assert units % num_heads == 0,\
            'In BERTEncoder, The units should be divided exactly ' \
            'by the number of heads. Received units={}, num_heads={}' \
            .format(units, num_heads)
        self._max_length = max_length
        self._units = units
        self._output_attention = output_attention
        self._output_all_encodings = output_all_encodings
        self._dropout = dropout
        self._layer_norm_eps = layer_norm_eps

        with self.name_scope():
            if dropout:
                self.dropout_layer = nn.Dropout(rate=dropout)
            self.layer_norm = nn.LayerNorm(in_channels=units, epsilon=self._layer_norm_eps)
            self.position_weight = self.params.get('position_weight', shape=(max_length, units),
                                                   init=weight_initializer)
            self.transformer_cells = nn.HybridSequential()
            for i in range(num_layers):
                cell = BERTEncoderCell(units=units, hidden_size=hidden_size, num_heads=num_heads,
                                       weight_initializer=weight_initializer,
                                       bias_initializer=bias_initializer, dropout=dropout,
                                       output_attention=output_attention,
                                       prefix='transformer%d_' % i, activation=activation,
                                       layer_norm_eps=layer_norm_eps, pre_norm=pre_norm)
                self.transformer_cells.add(cell)

    def __call__(self, inputs, states=None, valid_length=None):
        """Encode the inputs given the states and valid sequence length.

        Parameters
        ----------
        inputs : NDArray or Symbol
            Input sequence. Shape (batch_size, length, C_in)
        states : list of NDArrays or Symbols
            Initial states. The list of initial states and valid length for self attention
        valid_length : NDArray or Symbol
            Valid lengths of each sequence. This is usually used when part of sequence has
            been padded. Shape (batch_size,)

        Returns
        -------
        encoder_outputs: list
            Outputs of the encoder. Contains:

            - outputs of the transformer encoder. Shape (batch_size, length, C_out)
            - additional_outputs of all the transformer encoder
        """
        return super().__call__(inputs, states, valid_length)

    def hybrid_forward(self, F, inputs, states=None, valid_length=None, position_weight=None):
        # pylint: disable=arguments-differ
        """Encode the inputs given the states and valid sequence length.

        Parameters
        ----------
        inputs : NDArray or Symbol
            Input sequence. Shape (length, batch_size, C_in)
        states : list of NDArrays or Symbols
            Initial states. The list of initial states and valid length for self attention
        valid_length : NDArray or Symbol
            Valid lengths of each sequence. This is usually used when part of sequence has
            been padded. Shape (batch_size,)

        Returns
        -------
        outputs : NDArray or Symbol, or List[NDArray] or List[Symbol]
            If output_all_encodings flag is False, then the output of the last encoder.
            If output_all_encodings flag is True, then the list of all outputs of all encoders.
            In both cases, shape of the tensor(s) is/are (length, batch_size, C_out)
        additional_outputs : list
            Either be an empty list or contains the attention weights in this step.
            The attention weights will have shape (batch_size, length) or
            (batch_size, num_heads, length, length)

        """
        # axis 0 is for length
        steps = F.contrib.arange_like(inputs, axis=0)
        if valid_length is not None:
            zeros = F.zeros_like(steps)
            # valid_length for attention, shape = (batch_size, seq_length)
            attn_valid_len = F.broadcast_add(F.reshape(valid_length, shape=(-1, 1)),
                                             F.reshape(zeros, shape=(1, -1)))
            attn_valid_len = F.cast(attn_valid_len, dtype='int32')
            if states is None:
                states = [attn_valid_len]
            else:
                states.append(attn_valid_len)
        else:
            attn_valid_len = None

        if states is None:
            states = [steps]
        else:
            states.append(steps)

        # positional encoding
        positional_embed = F.Embedding(steps, position_weight, self._max_length, self._units)
        inputs = F.broadcast_add(inputs, F.expand_dims(positional_embed, axis=1))

        if self._dropout:
            inputs = self.dropout_layer(inputs)
        inputs = self.layer_norm(inputs)
        outputs = inputs

        all_encodings_outputs = []
        additional_outputs = []
        for cell in self.transformer_cells:
            outputs, attention_weights = cell(inputs, attn_valid_len)
            inputs = outputs
            if self._output_all_encodings:
                if valid_length is not None:
                    outputs = F.SequenceMask(outputs, sequence_length=valid_length,
                                             use_sequence_length=True, axis=0)
                all_encodings_outputs.append(outputs)

            if self._output_attention:
                additional_outputs.append(attention_weights)

        if valid_length is not None and not self._output_all_encodings:
            # if self._output_all_encodings, SequenceMask is already applied above
            outputs = F.SequenceMask(outputs, sequence_length=valid_length,
                                     use_sequence_length=True, axis=0)

        if self._output_all_encodings:
            return all_encodings_outputs, additional_outputs
        return outputs, additional_outputs


class BERTModel(HybridBlock):
    def __init__(self, encoder, vocab_size=None, token_type_vocab_size=None, units=None,
                 embed_size=None, embed_dropout=0.0, embed_initializer=None, word_embed=None,
                 token_type_embed=None, use_pooler=True, use_token_type_embed=True, prefix=None,
                 params=None):
        super(BERTModel, self).__init__(prefix=prefix, params=params)
        self._use_pooler = use_pooler
        self._use_token_type_embed = use_token_type_embed
        self._vocab_size = vocab_size
        self._units = units
        self.encoder = encoder
        # Construct word embedding
        self.word_embed = self._get_embed(word_embed, vocab_size, embed_size, embed_initializer,
                                          embed_dropout, 'word_embed_')
        # Construct token type embedding
        if use_token_type_embed:
            self.token_type_embed = self._get_embed(token_type_embed, token_type_vocab_size,
                                                    embed_size, embed_initializer, embed_dropout,
                                                    'token_type_embed_')
        if self._use_pooler:
            # Construct pooler
            self.pooler = self._get_pooler(units, 'pooler_')

    def _get_embed(self, embed, vocab_size, embed_size, initializer, dropout, prefix):
        """ Construct an embedding block. """
        if embed is None:
            assert embed_size is not None, '"embed_size" cannot be None if "word_embed" or ' \
                                           'token_type_embed is not given.'
            with self.name_scope():
                embed = nn.HybridSequential(prefix=prefix)
                with embed.name_scope():
                    embed.add(
                        nn.Embedding(input_dim=vocab_size, output_dim=embed_size,
                                     weight_initializer=initializer))
                    if dropout:
                        embed.add(nn.Dropout(rate=dropout))
        assert isinstance(embed, HybridBlock)
        return embed

    def _get_pooler(self, units, prefix):
        """ Construct pooler.

        The pooler slices and projects the hidden output of first token
        in the sequence for segment level classification.

        """
        with self.name_scope():
            pooler = nn.Dense(units=units, flatten=False, activation='tanh', prefix=prefix)
        return pooler

    def __call__(self, inputs, token_types, valid_length=None, masked_positions=None):
        # pylint: disable=dangerous-default-value, arguments-differ
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a BERT model.
        """
        # XXX Temporary hack for hybridization as hybridblock does not support None inputs
        valid_length = [] if valid_length is None else valid_length
        masked_positions = [] if masked_positions is None else masked_positions
        return super(BERTModel, self).__call__(inputs, token_types, valid_length, masked_positions)

    def hybrid_forward(self, F, inputs, token_types, valid_length=None, masked_positions=None):
        # pylint: disable=arguments-differ
        """Generate the representation given the inputs.

        This is used in training or fine-tuning a BERT model.
        """
        # XXX Temporary hack for hybridization as hybridblock does not support None
        if isinstance(masked_positions, list) and len(masked_positions) == 0:
            masked_positions = None

        outputs = []
        seq_out, attention_out = self._encode_sequence(inputs, token_types, valid_length)
        outputs.append(seq_out)

        if self.encoder._output_all_encodings:
            assert isinstance(seq_out, list)
            output = seq_out[-1]
        else:
            output = seq_out

        if attention_out:
            outputs.append(attention_out)

        if self._use_pooler:
            pooled_out = self._apply_pooling(output)
            outputs.append(pooled_out)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    def _encode_sequence(self, inputs, token_types, valid_length=None):
        """Generate the representation given the input sequences.

        This is used for pre-training or fine-tuning a BERT model.
        """
        # embedding
        embedding = self.word_embed(inputs)
        if self._use_token_type_embed:
            type_embedding = self.token_type_embed(token_types)
            embedding = embedding + type_embedding
        # encoding
        outputs, additional_outputs = self.encoder(embedding, valid_length=valid_length)
        return outputs, additional_outputs

    def _apply_pooling(self, sequence):
        """Generate the representation given the inputs.

        This is used for pre-training or fine-tuning a BERT model.
        """
        outputs = sequence.slice(begin=(0, 0, 0), end=(None, 1, None))
        outputs = outputs.reshape(shape=(-1, self._units))
        return self.pooler(outputs)

    def _arange_like(self, F, inputs):
        """Helper function to generate int32 indices of a range"""
        inputs = inputs.reshape(-1)
        if F == mx.ndarray:
            seq_len = inputs.shape[0]
            arange = F.arange(seq_len, dtype=inputs.dtype, ctx=inputs.context)
        else:
            zeros = F.zeros_like(inputs)
            arange = F.arange(start=0, repeat=1, step=1, infer_range=True, dtype='int32')
            arange = F.elemwise_add(arange, zeros)
        return arange

    def _decode(self, F, sequence, masked_positions):
        """Generate unnormalized prediction for the masked language model task.

        This is only used for pre-training the BERT model.

        Inputs:
            - **sequence**: input tensor of sequence encodings.
              Shape (batch_size, seq_length, units).
            - **masked_positions**: input tensor of position of tokens for masked LM decoding.
              Shape (batch_size, num_masked_positions). For each sample in the batch, the values
              in this tensor must not be out of bound considering the length of the sequence.

        Outputs:
            - **masked_lm_outputs**: output tensor of token predictions for target masked_positions.
                Shape (batch_size, num_masked_positions, vocab_size).
        """
        masked_positions = masked_positions.astype('int32')
        mask_shape = masked_positions.shape_array()
        num_masked_positions = mask_shape.slice(begin=(1, ), end=(2, )).astype('int32')
        idx_arange = self._arange_like(F, masked_positions)
        batch_idx = F.broadcast_div(idx_arange, num_masked_positions)
        # batch_idx_1d =        [0,0,0,1,1,1,2,2,2...]
        # masked_positions_1d = [1,2,4,0,3,4,2,3,5...]
        batch_idx_1d = batch_idx.reshape((1, -1))
        masked_positions_1d = masked_positions.reshape((1, -1))
        position_idx = F.concat(batch_idx_1d, masked_positions_1d, dim=0)
        encoded = F.gather_nd(sequence, position_idx)
        encoded = encoded.reshape_like(masked_positions, lhs_begin=-2, lhs_end=-1, rhs_begin=0)
        decoded = self.decoder(encoded)
        return decoded
