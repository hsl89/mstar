import pytest
from mstar import AutoModel, AutoTokenizer

@pytest.mark.parametrize('use_cpu', [False, True])
def test_atm_20b_multi_device_inference(use_cpu):
    tokenizer = AutoTokenizer.from_pretrained('atm-PreLNSeq2Seq-20B')
    model = AutoModel.from_pretrained('atm-PreLNSeq2Seq-20B')

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
