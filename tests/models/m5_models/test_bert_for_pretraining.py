import torch
import pytest
import numpy as np

from mstar.models.m5_models import M5BertConfig, M5BertForPreTrainingPreLN
from mstar.models.m5_models.bert import BertEmbeddings, M5BertModel
from mstar import AutoModel, AutoTokenizer

BERT_CONFIG_DICT = {
    "vocab_size": 32,
    "hidden_size": 4,
    "num_hidden_layers": 1,
    "num_attention_heads": 1,
    "intermediate_size": 64,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 32,
    "initializer_range": 0.02,
    "type_vocab_size": 1,
    "output_transform": "pool",
    "use_fused_softmax": True,
}

def test_bert_for_pretraining_pre_ln_output():
    bert_config = M5BertConfig(**BERT_CONFIG_DICT)
    batch_size = 4
    input_ids = torch.randint(
        0,
        bert_config.vocab_size,
        (batch_size, bert_config.max_position_embeddings),
    ).to('cuda:0')
    valid_len = torch.tensor([bert_config.max_position_embeddings] * batch_size).to('cuda:0')

    # construct expected outputs by initializing BERT with no output_transform
    torch.manual_seed(100)
    np.random.seed(100)
    bert = M5BertForPreTrainingPreLN(bert_config).to('cuda:0').eval()
    with torch.no_grad():
        output = bert(input_ids, valid_len, embedding_mode=True)
    assert output.shape == torch.Size([batch_size, bert_config.hidden_size])


def test_mstar_bert_model_loading():
    model = AutoModel.from_pretrained("mstar-bert-5B-bedrock", revision="20221004-300B-MLMonly").to('cpu').to(dtype=torch.bfloat16).eval()
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    customer_text = "hello world"
    output = tokenizer(customer_text, max_length=512, truncation=True, return_tensors="pt")
    text_encoding = output['input_ids']
    valid_length = output['attention_mask'].sum(axis=1)
