import os
from mstar.AutoTokenizer import from_pretrained as tok_from_pretrained
from mstar.AutoModel import from_pretrained as model_from_pretrained
from mstar.utils.hf_utils import get_model_file_from_s3, get_md5sum

def test_get_mstar_tokenizer_file_from_s3():
    """Functional test of loading tokenizer from s3 bucket."""
    tokenizer = tok_from_pretrained("mstar-gpt2-600M")
    res = tokenizer("hello world")
    assert res == {'input_ids': [51201, 6295, 2609], 'attention_mask': [1, 1, 1]}
    
    
def test_load_model_file_from_s3():
    """Functional test of loading model from s3 bucket."""
    key = "mstar-bert-tiny"
    revision = "test"
    downloaded_folder = get_model_file_from_s3(key, revision=revision, force_download=True)
    model = model_from_pretrained(downloaded_folder) 
    assert model.config.architectures == ["BertModel"]
    

def test_get_model_file_from_s3():
    """Functional test of downloading model files from s3."""
    key = "mstar-bert-tiny"
    revision = "test"
    downloaded_folder = get_model_file_from_s3(key, revision=revision, force_download=True)
    assert "9f8d20dc729dc0f3c05fbefc6ca4e8fa" == get_md5sum(downloaded_folder+"/pytorch_model.bin")
    assert "78d21f863d09a1b64b1f9921727454e2" == get_md5sum(downloaded_folder+"/config.json")
    assert os.path.exists(downloaded_folder+"/config.json")
    assert os.path.exists(downloaded_folder+"/pytorch_model.bin")
    
