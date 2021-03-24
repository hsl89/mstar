import torch as th
import numpy as np
import pytest
from mstar.models.transformer import TransformerDecoderLayer, TransformerDecoder


@pytest.mark.parametrize('layout', ['NT', 'TN'])
@pytest.mark.parametrize('units,mem_units', [(8, 8), (16, 16)])
@pytest.mark.parametrize('seq_length,mem_length', [(50, 30), (30, 5)])
def test_transformer_decoder_layer(layout, units, mem_units, seq_length, mem_length):
    layer = TransformerDecoderLayer(units=units, mem_units=mem_units,
                                    hidden_size=128, num_heads=4,
                                    layout=layout)
    layer.eval()
    batch_size = 5
    units = layer.units
    if layout == 'TN':
        data = th.normal(0, 1, (seq_length, batch_size, units))
        mem = th.normal(0, 1, (mem_length, batch_size, units))
    else:
        data = th.normal(0, 1, (batch_size, seq_length, units))
        mem = th.normal(0, 1, (batch_size, mem_length, units))
    mem_valid_length = th.tensor([1, 2, 3, 2, 2])
    mem_attn_mask = th.ones(batch_size, seq_length, mem_length)
    for i, valid_length in enumerate(mem_valid_length):
        mem_attn_mask[i, :, valid_length:] = 0
    self_causal_mask = th.tril(th.ones((batch_size, seq_length, seq_length)), 0)
    with th.no_grad():
        out = layer(data, mem, self_causal_mask, mem_attn_mask)
        init_states = layer.init_states(batch_size=batch_size,
                                        device=data.device)
        seq_out_by_inc_decode = []
        states = init_states
        if layout == 'TN':
            for i in range(seq_length):
                step_out, states = layer.incremental_decode(data[i, :, :], states, mem,
                                                            mem_valid_length,
                                                            mem_attn_mask[:, i:(i + 1), :])
                seq_out_by_inc_decode.append(step_out)
            seq_out_by_inc_decode = th.stack(seq_out_by_inc_decode, 0)
        else:
            for i in range(seq_length):
                step_out, states = layer.incremental_decode(data[:, i, :], states, mem,
                                                            mem_valid_length,
                                                            mem_attn_mask[:, i:(i + 1), :])
                seq_out_by_inc_decode.append(step_out)
            seq_out_by_inc_decode = th.stack(seq_out_by_inc_decode, 1)
        np.testing.assert_allclose(seq_out_by_inc_decode.detach().cpu().numpy(),
                                   out.detach().cpu().numpy(), 1E-4, 1E-4)


@pytest.mark.parametrize('layout', ['NT', 'TN'])
@pytest.mark.parametrize('units,mem_units', [(8, 8)])
@pytest.mark.parametrize('seq_length,mem_length', [(10, 30), (3, 10)])
@pytest.mark.parametrize('recurrent', [False, True])
@pytest.mark.parametrize('pre_norm', [False, True])
def test_transformer_decoder(layout, units, mem_units, seq_length,
                             mem_length, recurrent, pre_norm):
    model = TransformerDecoder(units=units, mem_units=mem_units,
                               hidden_size=64, num_heads=4,
                               num_layers=3,
                               layout=layout, recurrent=recurrent,
                               pre_norm=pre_norm)
    model.eval()
    batch_size = 5
    units = model.units
    if layout == 'TN':
        data = th.normal(0, 1, (seq_length, batch_size, units))
        mem = th.normal(0, 1, (mem_length, batch_size, units))
    else:
        data = th.normal(0, 1, (batch_size, seq_length, units))
        mem = th.normal(0, 1, (batch_size, mem_length, units))
    mem_valid_length = th.tensor([1, 2, 3, 2, 2])
    data_valid_length = th.randint(1, seq_length, (batch_size,))
    with th.no_grad():
        out = model(data, data_valid_length, mem, mem_valid_length)
        init_states = model.init_states(batch_size=batch_size, device=data.device)
        seq_out_by_inc_decode = []
        states = init_states
        if layout == 'TN':
            for i in range(seq_length):
                step_out, states = model.incremental_decode(data[i, :, :], states, mem,
                                                            mem_valid_length)
                seq_out_by_inc_decode.append(step_out)
            seq_out_by_inc_decode = th.stack(seq_out_by_inc_decode, 0)
            for i in range(batch_size):
                batch_out_by_inc = seq_out_by_inc_decode[:data_valid_length[i], i, :].cpu().numpy()
                batch_out = out[:data_valid_length[i], i, :].cpu().numpy()
                np.testing.assert_allclose(batch_out_by_inc, batch_out, 1E-4, 1E-4)
        else:
            for i in range(seq_length):
                step_out, states = model.incremental_decode(data[:, i, :], states, mem,
                                                            mem_valid_length)
                seq_out_by_inc_decode.append(step_out)
            seq_out_by_inc_decode = th.stack(seq_out_by_inc_decode, 1)
            for i in range(batch_size):
                batch_out_by_inc = seq_out_by_inc_decode[i, :data_valid_length[i], :].cpu().numpy()
                batch_out = out[i, :data_valid_length[i], :].cpu().numpy()
                np.testing.assert_allclose(batch_out_by_inc, batch_out, 1E-4, 1E-4)
