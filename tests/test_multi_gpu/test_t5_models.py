import torch
from mstar import AutoModel, AutoTokenizer

def test_t5_1_9B_multi_device_inference():
    model_name = "mstar-t5-1-9B-tracer"
    length = 512
    device = "cuda:0"

    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.parallelize()

    text_in = "<extra_id_0> Amazon is a store that "
    tokens_in = tokenizer(text_in, return_tensors = "pt")
    tokens_in.to(device)
    input_ids_in = tokens_in["input_ids"]
    attention_mask_in = tokens_in["attention_mask"]

    tokens_out = model.generate(input_ids = input_ids_in, attention_mask = attention_mask_in, min_length = length, max_length = length)
    text_out = tokenizer.batch_decode(tokens_out)
    print(text_out)
