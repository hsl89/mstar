# we decouple the pipeline's capability to do inference and everything else (data loading, model loading, integration with MLflow etc)
# the reason is that inference is the most costly step, once its correctness is verifed,
# we can run integration test by using fake inference, i.e. generating random vectors in the encoding stage
# in test.py, we only test our pipeline's capability to do distributed inference
# see integ_test.sh for integration test

from omegaconf import OmegaConf, DictConfig
from typing import List
import os
import json
import time
from loguru import logger
import fire

import torch
import torch.multiprocessing as mp

from main import setup, cleanup
import model_class
import load_models
import inference
import utils


dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, "config.json"), "r") as f:
    config = json.load(f)

TEST_SENTENCES = os.path.join(dir_path, config["test_sentences"])
LOG_PATH = config["log_path"]
logger.add(LOG_PATH, format="{time} {level} {message}", level="DEBUG")


def inference_step(rank: int, world_size: int, sentences: List[str], cfg: DictConfig):
    if world_size > 1:
        setup(rank, world_size)

    load_model_fn = getattr(load_models, cfg.model.load_model_fn)
    inference_fn = getattr(inference, cfg.model.inference_fn)

    model = getattr(model_class, cfg.model.model_class)(
        rank=rank,
        world_size=world_size,
        load_model_fn=load_model_fn,
        inference_fn=inference_fn,
        debug=cfg.debug,
        use_ddp=cfg.use_ddp,
        task_params=cfg.task,
        model_params=cfg.model.model_params,
        tokenizer_params=cfg.model.tokenizer_params,
        data_params=cfg.model.data_params,
    )

    for _ in range(2):
        vectors = model.encode(sentences)
        assert len(vectors) == len(sentences)
    return


def create_config(model_type):
    """
    create config for the model to be test
    model_type: choice of mstar and hf
    """
    confs = {
        "hf": OmegaConf.load(
            os.path.join(dir_path, "conf", "model", "distilbert.yaml")
        ),
        "mstar": OmegaConf.load(os.path.join(dir_path, "conf", "model", "mstar.yaml")),
        "sgpt": OmegaConf.load(os.path.join(dir_path, "conf", "model", "sgpt.yaml")),
        "sgpt_specb": OmegaConf.load(os.path.join(dir_path, "conf", "model", "sgpt_specb.yaml")),
    }

    task = OmegaConf.load(os.path.join(dir_path, "conf", "task", "retrieval.yaml"))

    if model_type not in confs:
        raise ValueError("model_type: %s not supported" % (model_type, ))

    model = conf = confs[model_type]
    return OmegaConf.create(
        {
            "use_ddp": True,
            "debug": False,
            "exp_name": "null",
            "split": ["test"],
            "model": model,
            "task": task,
        }
    )


def test_inference(model_type="mstar"):
    cfg = create_config(model_type)
    sentences = utils.read_jsonl(TEST_SENTENCES)
    sentences = [x["text"] for x in sentences]
    world_size = torch.cuda.device_count()
    ctx = mp.get_context("spawn")
    procs = []
    for i in range(world_size):
        p = ctx.Process(
            target=inference_step,
            kwargs={
                "rank": i,
                "world_size": world_size,
                "sentences": sentences,
                "cfg": cfg,
            },
        )
        p.start()
        procs.append((p, time.time()))

    for i, (p, s) in enumerate(procs):
        logger.info("***** waiting to process %s to finish *****\n\n" % i)
        p.join()
        e = time.time()
        logger.info("***** process %s finished in %0.2f seconds *****\n\n" % (i, e - s))
    return


def test_average_over_languages():
    res = {
        "lang1" : {
            "d1": 0.5,
            "d2": 0.3
        },
        "lang2" : {
            "d1": 0.4, 
            "d2": 0.5
        }
    }
    out = utils.average_over_languages(res, ["lang1", "lang2"])
    assert out == {"d1": 0.45, "d2": 0.4}

    
def main(fn, **kwargs):
    globals()[fn](**kwargs)
    return


if __name__ == "__main__":
    fire.Fire(main)
