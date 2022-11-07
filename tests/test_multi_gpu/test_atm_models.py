import os
import pytest
import shutil
from mstar import AutoModel, AutoTokenizer

mstar_cache_home = os.path.expanduser(
    os.getenv(
        "MSTAR_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "mstar")
    )
)

@pytest.mark.parametrize('use_cpu', [False, True])
def test_atm_20b_multi_device_inference(use_cpu):

    key = 'atm-PreLNSeq2Seq-20B'
    revision = 'main'

    tokenizer = AutoTokenizer.from_pretrained(key, revision=revision)
    model = AutoModel.from_pretrained(key, revision=revision)

    model.bfloat16()
    ## Load on 1 CPU and 3 GPUs or 4 GPUs 
    ## (at least 4 devices on g4dn.12xlarge, otherwise there will be OOM issue)
    model.parallelize(num_devices=4, start_device=0, use_cpu=use_cpu)

    test = """[CLM] Question: Who is the vocalist of coldplay? Answer:"""
    encoded = tokenizer(test, return_tensors="pt")
    if not use_cpu:
        encoded = encoded.to('cuda:0')
    generated_tokens = model.generate(input_ids=encoded['input_ids'],
                                      max_length=32,
                                      num_beams=3,
                                      num_return_sequences=3,
                                      bad_words_ids=[[0], [2012, 2006]],
                                      early_stopping=True)
    print(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))

    
    # clean up model download to save space
    # may want this to only happen after the 
    # last test to avoid double download
    path_for_cleanup = os.path.join(mstar_cache_home, "transformers", key,revision)
    shutil.rmtree(path_for_cleanup)
