import pytest
from mstar import AutoTokenizer
from mstar.tokenizers.m5_tokenizers import M5SentencepieceTokenizer


def test_sentencepiece_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("m5-Bert-5B-20220913")
    tokens = tokenizer.tokenize("Hello, my dog is the cutest.")

    print(tokens)
    assert tokens == ["▁hello", ",", "▁my", "▁dog", "▁is", "▁the", "▁cutest", "."]

    assert M5SentencepieceTokenizer.is_first_subword("▁Am")
    assert not M5SentencepieceTokenizer.is_first_subword("Am")


@pytest.mark.parametrize(
    "model_id, spl_token_list, expected_size_increase",
    [
        ("m5-Bert-5B-20220913", ["[TEST]"], 1),
        ("m5-Bert-5B-20220913", None, 0),
        ("m5-Bert-5B-20220913", ["[TEST1]", "[TEST2]"], 2),
        ("m5-Bert-5B-20220913", ["[TEST]", "[CLS]"], 1),
        ("m5-Bert-5B-20220913", ["[CLS]"], 0),
    ],
)
def test_m5sentencepiece_tokenizer_with_special_tokens(
    model_id, spl_token_list, expected_size_increase
):
    tokenizer = AutoTokenizer.from_pretrained(model_id, additional_special_tokens=spl_token_list)

    test_string = "My dog is the cutest."
    expected_tokenized_output = ["▁my", "▁dog", "▁is", "▁the", "▁cutest", "."]
    if spl_token_list:
        for spl_token in spl_token_list:
            test_string += spl_token
            expected_tokenized_output.append(spl_token)

    tokens = tokenizer.tokenize(test_string)

    assert tokens == expected_tokenized_output
    assert (tokenizer.vocab_size + expected_size_increase) == tokenizer.full_vocab_size


@pytest.mark.parametrize(
    "model_id, test_input",
    [
        ("m5-Bert-5B-20220913", "Jeden Tag geräucherter Bauernschinken"),
        ("m5-Bert-5B-20220913", "り発売日が延期"),
        ("m5-Bert-5B-20220913", "Kontrol Göstergeli Ölçü"),
        ("m5-Bert-5B-20220913", "hello 123, abc"),
        ("m5-Bert-5B-20220913", "5678, 890"),
        ("m5-Bert-5B-20220913", "Göstergeli, 1 Ölçü")
    ],
)
def test_m5sentencepiece_tokenizer_with_special_char_input(
    model_id, test_input
):
    import sentencepiece as spm
    cache_dir = "/tmp"
    m5_tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    sp_tokenizer = spm.SentencePieceProcessor()
    sp_tokenizer.load(cache_dir + "/tokenizers/" + model_id + "/main/spiece.model")

    m5_tokenized_text = m5_tokenizer.tokenize(test_input)
    spm_tokenized_text = sp_tokenizer.encode_as_pieces(test_input)
    sp_tokenizer = None

    # verify that the new m5 tokenizer produces the same output as spm
    assert m5_tokenized_text == spm_tokenized_text

@pytest.mark.parametrize(
    "model_id, vocab_size",
    [
        ("m5-Bert-5B-20220913", 512035),
        ("m5-Bert-50B-20220913", 256032)
    ],
)
def test_vocab_size(model_id, vocab_size):
    special_tokens = [
        "[MASK]","[SEP]","[CLS]","[en_CA]","[mr_IN]","[pl_PL]","[sv_SE]","[en_US]","[it_IT]","[ta_IN]",
        "[te_IN]","[ml_IN]","[nl_NL]","[en_AU]","[en_IN]","[hi_IN]","[ko_KR]","[de_DE]","[pt_PT]","[ja_JP]",
        "[zh_CN]","[pt_BR]","[en_AE]","[zh_TW]","[fr_FR]","[kn_IN]","[es_MX]","[ar_AE]","[cs_CZ]","[en_GB]",
        "[en_SG]","[he_IL]","[tr_TR]","[ru_RU]","[es_ES]"
    ]
    m5_tokenizer = AutoTokenizer.from_pretrained(model_id, additional_special_tokens=special_tokens)
    assert len(m5_tokenizer) == vocab_size
