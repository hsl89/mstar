import torch
from mstar import AutoModel
from accelerate import infer_auto_device_map

def test_auto_model_device_map():
    ## Test for huggingface models with infer_auto_device_map
    gpt2_large = AutoModel.from_pretrained("gpt2-large")
    device_map = infer_auto_device_map(gpt2_large, max_memory={0: "1GiB", 1: "1GiB", 2: "1GiB", 3: "1GiB"})
    gpt2_large = AutoModel.from_pretrained("gpt2-large", device_map=device_map)
    assert gpt2_large.hf_device_map['wte'] == 0
    assert gpt2_large.hf_device_map['h.9.mlp.dropout'] == 1
    assert gpt2_large.hf_device_map['h.34'] == 2

    ## Test for mstar models gpt2 with infer_auto_device_map
    mstar_gpt2 = AutoModel.from_pretrained("mstar-gpt2LMHead-1.3B-easel-c1024-t417B-b0.92M-06102022")
    device_map = infer_auto_device_map(mstar_gpt2, max_memory={0: "2GiB", 1: "2GiB", 2: "2GiB", 3: "2GiB"})
    mstar_gpt2 = AutoModel.from_pretrained("mstar-gpt2LMHead-1.3B-easel-c1024-t417B-b0.92M-06102022", device_map=device_map)
    assert mstar_gpt2.hf_device_map['transformer.drop'] == 0
    assert mstar_gpt2.hf_device_map['transformer.h.11'] == 1
    assert mstar_gpt2.hf_device_map['transformer.h.23'] == 2

    ## Test for mstar models gpt2 with balanced strategy
    ## which will split the model evenly across GPUs
    mstar_gpt2 = AutoModel.from_pretrained("mstar-gpt2LMHead-1.3B-easel-c1024-t417B-b0.92M-06102022", device_map="balanced")
    assert len(set(mstar_gpt2.hf_device_map.values())) == torch.cuda.device_count()

    ## Test for mstar models t5 with auto strategy
    ## which is default to balanced and will split the model evenly across GPUs
    mstar_t5 = AutoModel.from_pretrained("mstar-t5-1-9B-bedrock-alexatm", device_map="auto")
    assert len(set(mstar_t5.hf_device_map.values())) == torch.cuda.device_count()

    ## Test for mstar models t5 with balanced_low_0 strategy
    ## which will split the model evenly across GPUs while leaving the most available memory on GPU 0
    mstar_t5 = AutoModel.from_pretrained("mstar-t5-1-9B-bedrock-alexatm", device_map="balanced_low_0")
    assert len(set(mstar_t5.hf_device_map.values())) == torch.cuda.device_count() - 1

    ## Test for mstar models gpt2-6.7B with sequential strategy
    ## which corresponds to the previous auto: fill each GPU sequentially
    ## (and if the user has lots of GPU spaces, some are not used at all)
    mstar_gpt2_6B = AutoModel.from_pretrained("mstar-gpt2LMHead-6.7B", device_map="sequential")
    assert mstar_gpt2_6B.hf_device_map['transformer.wpe'] == 0
    assert mstar_gpt2_6B.hf_device_map['transformer.h.28'] == 'cpu'
