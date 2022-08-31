import unittest
import torch
from mstar.models.atm_models import PreLnConfig, PreLnForMaskedLM, PreLNSeq2SeqConfig, PreLNSeq2SeqModel

class ATMModelsTest(unittest.TestCase):

    def test_preln(self):
        config = PreLnConfig(
            hidden_size=64,
            intermediate_size=128,
            max_position_embeddings=16,
            num_attention_heads=8,
            num_hidden_layers=2,
            vocab_size=10,
            type_vocab_size=1
        )
        model = PreLnForMaskedLM(config)

        fake_data = torch.tensor([[1, 2, 3, 4]])
        attn_mask = torch.ones_like(fake_data)

        output = model(fake_data, attn_mask)

        self.assertEqual(output.logits.shape[0], 1)
        self.assertEqual(output.logits.shape[1], 4)
        self.assertEqual(output.logits.shape[2], 10)


    def test_preln_seq2seq(self):
        config = PreLNSeq2SeqConfig(
            add_bias_logits=False,
            add_final_layer_norm=False,
            bos_token_id=1,
            eos_token_id=2,
            d_model=128,
            decoder_attention_heads=16,
            decoder_ffn_dim=128,
            decoder_layers=2,
            decoder_start_token_id=1,
            do_blenderbot_90_layernorm=False,
            encoder_attention_heads=16,
            encoder_ffn_dim=128,
            encoder_layers=2,
            extra_pos_embeddings=2,
            force_bos_token_to_be_generated=False,
            forced_eos_token_id=2,
            is_encoder_decoder=True,
            max_position_embeddings=10,
            share_embeddings=True,
            type_vocab_size=1,
            vocab_size=10
        )
        model = PreLNSeq2SeqModel(config)

        fake_data = torch.tensor([[1, 2, 3, 4]])
        attn_mask = torch.ones_like(fake_data)

        output = model(fake_data, attn_mask)

        self.assertEqual(output.last_hidden_state.shape[0], 1)
        self.assertEqual(output.last_hidden_state.shape[1], 4)
        self.assertEqual(output.last_hidden_state.shape[2], 128)

        self.assertEqual(output.encoder_last_hidden_state.shape[0], 1)
        self.assertEqual(output.encoder_last_hidden_state.shape[1], 4)
        self.assertEqual(output.encoder_last_hidden_state.shape[2], 128)