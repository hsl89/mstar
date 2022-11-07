import shutil
import os
import torch
import torch.multiprocessing as mp
from mstar import AutoModel
from accelerate import infer_auto_device_map
from mstar.AutoTokenizer import from_pretrained as tok_from_pretrained
from mstar.AutoModel import from_pretrained as model_from_pretrained

mstar_cache_home = os.path.expanduser(
    os.getenv(
        "MSTAR_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "mstar")
    )
)

def worker1(rank):
    torch.cuda.set_device(rank)
    tokenizer = tok_from_pretrained("atm-PreLNSeq2Seq-5B")
    print(tokenizer("hello world"))


def test_tokenizer_concurrent_write():
    mp.spawn(worker1, nprocs=4, args=())


def worker2(rank):
    torch.cuda.set_device(rank)
    model = model_from_pretrained("mstar-t5-1-9B-bedrock-alexatm")


def test_model_concurrent_write():
    mp.spawn(worker2, nprocs=2, args=())


def test_auto_model_device_map():


    ## Test for mstar models gpt2 with infer_auto_device_map
    key = "mstar-gpt2LMHead-1.3B-easel-c1024-t417B-b0.92M-06102022"
    revision = "main"
    mstar_gpt2 = AutoModel.from_pretrained(key, revision=revision)
    device_map = infer_auto_device_map(mstar_gpt2, max_memory={0: "2GiB", 1: "2GiB", 2: "2GiB", 3: "2GiB"})
    mstar_gpt2 = AutoModel.from_pretrained(key, revision=revision, device_map=device_map)
    assert mstar_gpt2.hf_device_map['transformer.drop'] == 0
    assert mstar_gpt2.hf_device_map['transformer.h.11'] == 1
    assert mstar_gpt2.hf_device_map['transformer.h.23'] == 2

    ## Test for mstar models gpt2 with balanced strategy
    ## which will split the model evenly across GPUs
    mstar_gpt2 = AutoModel.from_pretrained(key, revision=revision, device_map="balanced")
    assert len(set(mstar_gpt2.hf_device_map.values())) == torch.cuda.device_count()

    path_for_cleanup = os.path.join(mstar_cache_home, "transformers", key,revision)
    shutil.rmtree(path_for_cleanup)

    ## Test for mstar models t5 with auto strategy
    ## which is default to balanced and will split the model evenly across GPUs
    
    key = "mstar-t5-1-9B-bedrock-alexatm"
    revision = "main"
    mstar_t5 = AutoModel.from_pretrained(key, revision=revision, device_map="auto")
    assert len(set(mstar_t5.hf_device_map.values())) == torch.cuda.device_count()

    ## Test for mstar models t5 with balanced_low_0 strategy
    ## which will split the model evenly across GPUs while leaving the most available memory on GPU 0
    mstar_t5 = AutoModel.from_pretrained(key, revision=revision, device_map="balanced_low_0")
    assert len(set(mstar_t5.hf_device_map.values())) == torch.cuda.device_count() - 1
    path_for_cleanup = os.path.join(mstar_cache_home, "transformers", key,revision)
    shutil.rmtree(path_for_cleanup)

    ## Test for mstar models gpt2-6.7B with sequential strategy
    ## which corresponds to the previous auto: fill each GPU sequentially
    ## (and if the user has lots of GPU spaces, some are not used at all)
    key = "mstar-gpt2LMHead-6.7B"
    revision = "main" 
    mstar_gpt2_6B = AutoModel.from_pretrained(key, revision=revision, device_map="sequential")
    assert mstar_gpt2_6B.hf_device_map['transformer.wpe'] == 0
    assert mstar_gpt2_6B.hf_device_map['transformer.h.28'] == 'cpu'
    path_for_cleanup = os.path.join(mstar_cache_home, "transformers", key,revision)
    shutil.rmtree(path_for_cleanup)
