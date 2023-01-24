from mteb import MTEB
import jsonlines
import fire
import os
import json
from typing import List
from loguru import logger
import sys
from functools import reduce

from flatten_dict import flatten, unflatten
from collections.abc import MutableMapping

def check_chunk_size(tasks: List[str]):
    """check sizes of each chunk of documents for a task"""
    ev = MTEB(tasks=tasks)
    for task in ev.tasks:
        task.load_data()

        for split in task.description.get("eval_splits", []):
            print("chunk sizes for %s split of %s" % (split, task.description["name"]))
            for chunk in task.dataset[split]:
                print(len(chunk["sentences"]))
    return


def read_jsonl(fpath: str):
    docs = []
    with jsonlines.open(fpath, "r") as reader:
        for s in reader:
            docs.append(s)
    return docs

def get_logger():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, "config.json"), "r") as f:
        config = json.load(f)
    LOG_PATH = config["log_path"]
    config = {
        "handlers": [
            {"sink": sys.stdout, "format": "{time} - {message}"},
            {"sink": LOG_PATH, "serialize": True},
        ],
        "extra": {"user": "someone"}
    }
    logger.configure(**config)
    return logger

        
def average_over_languages(res, langs):
    if not any(map(lambda x: x in res, langs)): return res
    flattened = []
    for lang in filter(lambda x: x in res, langs):
        flattened.append(flatten(res[lang]))

    def add_dict(d1, d2):
        d3 = {}
        for k in d1:
            assert isinstance(d1[k], float) and isinstance(d2[k], float), "%s field in d1: %s or d2: %s is not a float" % (k, d1, d2)
            d3[k] = d1[k] + d2[k]
        return d3
            
    res = reduce(lambda x, y : add_dict(x, y), flattened)
    for k in res:
        res[k] /= len(flattened)
    return unflatten(res)

    
def main(fn, **kwargs):
    globals()[fn](**kwargs)
    return


if __name__ == "__main__":
    fire.Fire(main)
