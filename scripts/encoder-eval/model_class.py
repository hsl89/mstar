## wrap the hugging face model to be used for DenseRetrievalExactSearch ##
import ipdb
import os
import functools

from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import json

import utils

# from mteb import DRESModel

from datasets import Dataset
from transformers import default_data_collator, DataCollatorWithPadding
from tokenizers import Tokenizer

import time
from typing import List, Union, Dict, Callable

from loguru import logger
import sys

import preprocessing

dir_path = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(dir_path, "config.json"), "r") as f:
    config = json.load(f)

TEST_SENTENCES = os.path.join(dir_path, config["test_sentences"])
LOG_PATH = config["log_path"]

logger = utils.get_logger()

def shard_docs(sentences: List[str], world_size: int):
    """shard input documents for each ddp process
    Retrun:
        List[List[str]]
    """
    sz = len(sentences) // world_size
    shard_szs = [sz] * world_size
    rmd = len(sentences) - sz * world_size
    shard_szs[0] += rmd

    shards = []
    cur_idx = 0
    for cur_size in shard_szs:
        shard = []
        for j in range(cur_idx, cur_idx + cur_size):
            shard.append(sentences[j])
            cur_idx += 1
        shards.append(shard)
    return shards


class EncoderModel:
    """wrapper of Hugging Face or Mstar model"""

    def __init__(
        self,
        rank: int,
        world_size: int,
        load_model_fn: Callable,
        inference_fn: Callable,
        use_ddp: bool,
        debug: bool,
        task_params,
        model_params,
        tokenizer_params,
        data_params,
    ):
        self.rank = rank
        self.world_size = world_size

        self.task_params = task_params
        self.load_model_fn = load_model_fn
        self.inference_fn = inference_fn
        self.debug = debug
        self.use_ddp = use_ddp
        self.model_params = model_params
        self.tokenizer_params = tokenizer_params
        self.data_params = data_params         

        # avoid race condition while downloading model weights from s3
        if self.rank == 0:
            self.model, self.tokenizer = self.load_model()
        dist.barrier()

        if not self.rank == 0:
            self.model, self.tokenizer = self.load_model()
        dist.barrier()

        # if self.model_params.use_bfloat16:
        #     self.model = self.model.bfloat16()

        if self.rank == 0:
            logger.info(
                "model's embedding dimension: %s" % self.model.config.hidden_size
            )

    def load_model(self):
        return self.load_model_fn(
            device=self.rank,
            model_name=self.model_params.model_name,
            debug=self.debug,
            tokenizer_name=self.tokenizer_params.tokenizer_name,
            use_bfloat16 = self.model_params.use_bfloat16
        )

    def cast_output_per_task_type(self, vectors: torch.Tensor):
        if self.task_params.task_type in ["clustering", "sts"]:
            vectors = vectors.to("cpu").numpy()
        elif self.task_params.task_type in ["retrieval"]:
            vectors = vectors.to(self.rank)
        else:
            raise ValueError(
                "Only support task type in [clustering, sts, retrieval], got %s"
                % self.task_params.task_type
            )
        return vectors

    def encode(self, sentences, **kwargs) -> Tensor:
        if self.debug:
            sentences = sentences[:128]
        doc_size = len(sentences)

        logger.info(
            "***** sharding input sentences in process: %s *****\n\n" % self.rank
        )
        data_shards = shard_docs(sentences, world_size=self.world_size)
        sentences = data_shards[self.rank]

        for pfn in self.data_params.get("preprocessing", []):
            sentences = getattr(preprocessing, pfn)(sentences)

        # TODO: handle the case with sentences being empty
        data_loader = DataLoader(
            sentences, batch_size=self.data_params.batch_size, shuffle=False
        )
        ns, embd_dim = len(sentences), self.model.config.hidden_size

        if ns == 0:
            logger.debug("sentences for process: %s is empty" % self.rank)

        vectors = torch.zeros((ns, embd_dim), dtype=torch.float32, device=self.rank)
        i = 0
        for bix, batch in enumerate(data_loader):
            try:
                start = time.time()
                tokenized_input = self.tokenizer(batch, **self.tokenizer_params.kwargs)
                v = self.inference_fn(
                    self.model,
                    bix,
                    tokenized_input,
                    rank=self.rank,
                    **self.model_params.inference_fn_kwargs
                )
                sz = v.shape[0]
                vectors[i : i + sz] = v
                i += sz
                end = time.time()

                if self.rank == 0 and (bix + 1) % 10 == 0:
                    logger.info(
                        "Fraction of sentences completed: %s"
                        % ((bix + 1) / len(data_loader))
                    )
                    logger.info("Time to process one batch: %0.3f " % (end - start))

            except Exception as e:
                logger.error(
                    "Rank: %s Exception encountered while doing inference: %s"
                    % (self.rank, e)
                )
                raise e
            
            # locate vectors with NaN
            # nan = torch.sum(torch.isnan(v), dim=1) > 0
            # if torch.sum(nan) > 0:
            #     logger.error("Found sentences that result in vector with NaN")
            #     for i, n in enumerate(nan):
            #         if n > 0: 
            #             logger.error("Sentence: %s" % (str(batch[i])))
            #             logger.error("Tokenized input: %s" % (tokenized_input))
            #             logger.error("Vector: %s" % str(v[i]))

        if not self.use_ddp:
            return self.cast_output_per_task_type(vectors)
        dist.barrier()
        tensor_list = [None] * self.world_size
        if self.world_size > 1:
            try:
                s = time.time()
                dist.all_gather_object(tensor_list, vectors.to("cpu"))
                e = time.time()
                logger.info(
                    "**** rank: %s, all gather completed in %0.2f seconds *****"
                    % (self.rank, e - s)
                )
            except Exception as e:
                logger.error(
                    "***** rank: %s. Exception encountered when gathering output embeddings from other processes: %s"
                    % (self.rank, str(e))
                )
                raise e
            tensor_list = [t.to(self.rank) for t in tensor_list]
            vectors = torch.cat(tensor_list, dim=0).to(self.rank)
            assert (
                len(vectors) == doc_size
            ), "vectors size and doc_size mismatch, vector size: %s doc_size: %s" % (
                str(vectors.shape),
                doc_size,
            )
        dist.barrier()
        return self.cast_output_per_task_type(vectors)

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs):
        if self.task_params.get("max_query_length", None) is not None:
            self.tokenizer_params.kwargs.max_length = self.task_params.max_query_length
        return self.encode(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):
        if self.task_params.get("max_document_length", None) is not None:
            self.tokenizer_params.kwargs.max_length = (
                self.task_params.max_document_length
            )

        if type(corpus) is dict:
            sentences = [
                (corpus["title"][i] + self.data_params.sep + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                (doc["title"] + self.data_params.sep + doc["text"]).strip()
                if "title" in doc
                else doc["text"].strip()
                for doc in corpus
            ]
        return self.encode(sentences, batch_size=batch_size, **kwargs)


class SGPT(EncoderModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert (
            "asymmetric_search" in self.model_params
        ), "Must specify whether to use asymmetric search or not"
        

        if self.model_params.asymmetric_search:
            self.tokenizer_params.kwargs.add_special_tokens = True
        else:
            self.tokenizer_params.kwargs.add_special_tokens = False
        
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs):
        if self.task_params.get("max_query_length", None) is not None:
            self.tokenizer_params.kwargs.max_length = self.task_params.max_query_length
        
        if self.model_params.asymmetric_search:
            # https://github.com/Muennighoff/sgpt#asymmetric-semantic-search-be
            # both [ and ] are in sgpt tokenizer's vocab, so we can wrap queries by [] and 
            # then tokenize 
            sentences = ["".join(["[", s, "]"]) for s in queries]
        
        return self.encode(queries, batch_size=batch_size, **kwargs)        

    def encode_corpus(self, corpus: List[str], batch_size: int, **kwargs):
        if self.task_params.get("max_document_length", None) is not None:
            self.tokenizer_params.kwargs.max_length = (
                self.task_params.max_document_length
            )

        if type(corpus) is dict:
            sentences = [
                (corpus["title"][i] + self.data_params.sep + corpus["text"][i]).strip()
                if "title" in corpus
                else corpus["text"][i].strip()
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                (doc["title"] + self.data_params.sep + doc["text"]).strip()
                if "title" in doc
                else doc["text"].strip()
                for doc in corpus
            ]
            
        if self.model_params.asymmetric_search:
            sentences = ["".join(["{", s, "}"]) for s in sentences]
        return self.encode(sentences, batch_size=batch_size, **kwargs)



class FakeInferenceModel(EncoderModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def encode(self, sentences: List[str], **kwargs):
        v = torch.rand((len(sentences), 1024), dtype=torch.float32, device=self.rank)
        return self.cast_output_per_task_type(v)
