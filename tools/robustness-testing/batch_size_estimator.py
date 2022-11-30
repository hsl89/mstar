import string
from tqdm import tqdm
from copy import deepcopy
from bisect import bisect
import random
import logging
import collections
import numpy as np
import lm_eval
from lm_eval.api.utils import set_seed, DEFAULT_SEED

logger = logging.getLogger(__name__)

def calculate_batchsize(
    model,
    tasks,
    num_fewshot=0,
    seed=DEFAULT_SEED,
) -> dict:
    """"""
    requests = get_requests(tasks, num_fewshot, seed)

    min_batch_size = float('inf')

    # Create list of possible batch sizes in increments of 16 then 32.
    BATCH_SIZE_CANDIDATES = list(sorted([1, 8, 16, 48] + [32*i for i in range(1, 1000)]))

    # Search for the most conservative batch size across each type of request
    for reqtype, reqs in requests.items():
        largest_proto_req = get_prototypical_request(model, reqtype, reqs)

        batch_size = search(model, reqtype, largest_proto_req, 1, BATCH_SIZE_CANDIDATES)

        min_batch_size = min(batch_size, min_batch_size)
    
    # Reduce batch size by 10% to avoid occasional OOM errors
    batchsize_90_percent = (min_batch_size * 9) // 10
    batchsize_90_percent_idx = bisect(BATCH_SIZE_CANDIDATES, batchsize_90_percent) - 1
    reduce_90_percent_idx = max(batchsize_90_percent_idx, 0)

    return BATCH_SIZE_CANDIDATES[reduce_90_percent_idx]

def get_prototypical_request(model, reqtype, reqs):
    """Creates the prototypical request that will be used to populate every batch."""
    sorted_reqs = sort_requests_by_length(model, reqtype, reqs)
        
    largest_proto_req = sorted_reqs[-1][-1]
    print("req_type:", reqtype)
    
    if reqtype == "greedy_until":
        largest_proto_req = extend_context_for_generation_request(model, largest_proto_req)
    
    return largest_proto_req

def extend_context_for_generation_request(model, largest_proto_req):
    """Extends the context by the maximum length the model is allowed to generate to simulate the worst case memory usage."""

    # TODO: Randomly sample tokens from the vocabulary instead of a random character for a better approximation of the maximum possible length.
    new_context = largest_proto_req.args[0] + "".join([random.choice(string.printable) for _ in range(model.max_length)])
    
    print("Extended Context length:", len(new_context), len(model.tok_encode(new_context)))
    
    largest_proto_req = lm_eval.api.request.Request(
        largest_proto_req.request_type, 
        (new_context, *deepcopy(largest_proto_req.args[1:])), 
        largest_proto_req.index
    )
    
    return largest_proto_req

def sort_requests_by_length(model, reqtype, reqs):
    """Sorts requests by length. Target lengths are included in the length for loglikelihood requests."""
    if reqtype == 'loglikelihood':
            # Include target lengths in the sort
        sorted_reqs = sorted(
                [
                    (
                        len(model.tok_encode(req.args[0])), 
                        len(model.tok_encode(req.args[1])), 
                        req
                    ) 
                    for req in reqs
                ],
                key=lambda x: (x[0], x[1])
            )
    else:
        # This block may not be necessary anymore since we are now extending the context by the maximum length the model is allowed to generate. Requires investigation and testing if it can be removed without adverse effects.
        sorted_reqs = sorted(
            [(len(model.tok_encode(req.args[0])), req) for req in reqs],
            key=lambda x: x[0]
        )
    
    return sorted_reqs

def get_requests(tasks, num_fewshot, seed):
    """Creates requests exactly in the same way as the lm-evaluation-harness. Extracted from `evaluator.evaluate()`."""

    rng = np.random.default_rng(seed)
    task_dict = {}
    for task in tasks:
        if task.has_validation_docs() is False and task.has_test_docs() is False:
            continue
        # Create unique keys for each task-template pair.
        task_name = lm_eval.tasks.get_registry_name_from_task(task)
        template_name = task.prompt_template.name if task.prompt_template else None
        key = lm_eval.tasks._get_task_template_key(task_name, template_name)
        task_dict[key] = task

    versions = collections.defaultdict(dict)
    requests = collections.defaultdict(list)
    requests_origin = collections.defaultdict(list)

    docs = {}

    # Build contexts and collect language model requests.
    for task_template_key, task in task_dict.items():
        task_docs = task.evaluation_docs()

        logger.info(f"\n» Assigning unique IDs to '{task_template_key}' docs")
        task_docs = task_docs.map(
            lambda ex, idx: {**ex, "doc_id": idx}, with_indices=True
        )

        logger.info(f"» Filtering invalid docs from '{task_template_key}'")
        task_docs = task_docs.filter(lambda d: not task.invalid_doc_for_prompt(d))
        task_docs = task_docs.shuffle(generator=rng)

        logger.info(f"» Constructing '{task_template_key}' contexts and requests")
        pbar_limit = len(task_docs)

        for doc_id, doc in enumerate(
            tqdm(task_docs, total=pbar_limit)
        ):
            docs[(task_template_key, doc_id)] = doc
            ctx, fewshotex_logging_info = task.fewshot_context(
                doc=doc,
                num_fewshot=num_fewshot,
                rng=rng,
            )

            args = {"num_fewshot": num_fewshot}
            reqs = task.construct_requests(doc, ctx, args)
            if not isinstance(reqs, (list, tuple)):
                reqs = [reqs]
            for i, req in enumerate(reqs):
                requests[req.request_type].append(req)
                # i: Index in requests for a single task instance
                # doc_id: Unique id that we can get back to a doc using `docs`
                requests_origin[req.request_type].append(
                    (i, task_template_key, doc, doc_id, fewshotex_logging_info)
                )
        # Store the task version.
        versions[task_template_key] = task.VERSION
    return requests


def test_run(model, reqtype, reqs):
    """Try running the model."""
    try:
        getattr(model, reqtype)([req.args for req in reqs])
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "CUBLAS_STATUS_NOT_INITIALIZED when calling" in str(e) or " CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgemm" in str(e):
            logger.info("Exceeded memory capacity")
            return False
        else:
            raise e
    return True
    

def search(model, reqtype, proto_req, prev_batch_size, batch_sizes):
    """Binary search for the largest possible batch size that doesn't cause an OOM error."""
    if not batch_sizes:
        return prev_batch_size
    
    mid = len(batch_sizes) // 2
    mid_batch_size = batch_sizes[mid]

    original_batch_size = model.batch_size
    
    # Create a batch of simulated requests
    # The harness performs example deduplication in the model call, so we append a number to the end of each prototype context to make them unique.
    # We also create a batch of examples that is double the batch size to simulate the scenario where one batch is loaded onto the GPU before the prior batch has been cleared. This addresses most of the OOM errors that were not caught by this estimator.
    simulated_requests = [lm_eval.api.request.Request(proto_req.request_type, (proto_req.args[0] + str(i), *deepcopy(proto_req.args[1:])), proto_req.index) for i in range(mid_batch_size*2)]

    print("simulated_requests size:", len(simulated_requests))

    model._batch_size = mid_batch_size

    print("model batch size:", model.batch_size)

    successful_run = test_run(model, reqtype, simulated_requests)
    model._batch_size = original_batch_size
    
    if successful_run:
        return max(mid_batch_size, search(model, reqtype, proto_req, mid_batch_size, batch_sizes[mid+1:]))
    else:
        return search(model, reqtype, proto_req, prev_batch_size, batch_sizes[:mid])
