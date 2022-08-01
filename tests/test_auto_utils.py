import os
from mstar.AutoTokenizer import from_pretrained as tok_from_pretrained
from mstar.AutoModel import from_pretrained as model_from_pretrained
from mstar.utils.hf_utils import get_model_file_from_s3, get_md5sum

mstar_cache_home = os.path.expanduser(
    os.getenv(
        "MSTAR_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "mstar")
    )
)


def test_get_mstar_tokenizer_file_from_s3():
    """Functional test of loading tokenizer from s3 bucket."""
    model_name = "mstar-gpt2-600M"

    def _assert(tokenizer, cache_dir):
        assert os.path.exists(
            os.path.join(
                cache_dir, "tokenizers", model_name, "main", "tokenizer_config.json"
            )
        )
        res = tokenizer("hello world")
        assert res == {"input_ids": [51201, 6295, 2609], "attention_mask": [1, 1, 1]}

    # with cache_dir
    cache_dir = "/tmp"
    tokenizer = tok_from_pretrained("mstar-gpt2-600M", cache_dir=cache_dir)
    _assert(tokenizer, cache_dir)

    # without cache_dir
    tokenizer = tok_from_pretrained("mstar-gpt2-600M")
    _assert(tokenizer, mstar_cache_home)


def test_load_model_file_from_s3():
    """Functional test of loading model from s3 bucket."""
    key = "mstar-bert-tiny"
    revision = "test"

    def _assert(downloaded_folder, cache_dir):
        assert downloaded_folder == os.path.join(
            cache_dir, "transformers", key, revision
        ) 
        assert os.path.exists(os.path.join(downloaded_folder, "config.json"))
        assert os.path.exists(os.path.join(downloaded_folder, "pytorch_model.bin"))
        model = model_from_pretrained(downloaded_folder)
        assert model.config.architectures == ["BertModel"]

    # with cache_dir
    cache_dir = "/tmp"
    downloaded_folder = get_model_file_from_s3(
        key, revision=revision, force_download=True, cache_dir=cache_dir
    )
    _assert(downloaded_folder, cache_dir)

    # without cache_dir
    downloaded_folder = get_model_file_from_s3(
        key, revision=revision, force_download=True
    )
    _assert(downloaded_folder, mstar_cache_home)


def test_get_model_file_from_s3():
    """Functional test of downloading model files from s3."""
    key = "mstar-bert-tiny"
    revision = "test"
    def _assert(downloaded_folder, cache_dir):
        # check downloaded_folder is in cache_dir
        assert downloaded_folder == os.path.join(
            cache_dir, "transformers", key, revision
        )

        assert "9f8d20dc729dc0f3c05fbefc6ca4e8fa" == get_md5sum(
            downloaded_folder + "/pytorch_model.bin"
        )
        assert "78d21f863d09a1b64b1f9921727454e2" == get_md5sum(
            downloaded_folder + "/config.json"
        )
        assert os.path.exists(os.path.join(downloaded_folder, "config.json"))
        assert os.path.exists(os.path.join(downloaded_folder, "pytorch_model.bin"))

    # with cache_dir
    cache_dir="/tmp"
    downloaded_folder = get_model_file_from_s3(
        key, revision=revision, force_download=True, cache_dir="/tmp"
    )
    _assert(downloaded_folder, cache_dir)

    # without cache_dir
    downloaded_folder = get_model_file_from_s3(
        key, revision=revision, force_download=True
    )
    _assert(downloaded_folder, mstar_cache_home)

def test_auto_tokenizer_save_load():
    """Functional test of loading tokenizer from s3 bucket, saving to local and loading from local"""
    local_path = "./saved_tokenizer"
    tokenizer = tok_from_pretrained("atm-PreLNSeq2Seq-5B")
    tokenizer.save_pretrained(local_path)
    tokenizer2 = tok_from_pretrained(local_path)
    assert tokenizer("hello world") == tokenizer2("hello world")
