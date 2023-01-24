# use multiprocessing to evalute relevance score for retrieval tasks

import pytrec_eval
import json
import multiprocessing as mp
import random
import time
import psutil


def create_qrel_results(num_queries, num_docs):
    start = time.time()
    qrel_docs = {"d%s" % d: random.randint(0, 1) for d in range(num_docs)}
    res_docs = {"d%s" % d: random.uniform(-1.0, 1.0) for d in range(num_docs)}

    qrel = {}
    result = {}
    for i in range(num_queries):
        qrel["q%s" % i] = qrel_docs
        result["q%s" % i] = res_docs

    end = time.time()
    print("Time taken to retrieve: %s" % (end - start))
    return qrel, result


def run_eval(rank, qrel, result, output_queue=None):
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {"map", "ndcg"})
    score = evaluator.evaluate(result)
    if output_queue is not None:
        output_queue.put(score)
    return score


def test():
    qrel, result = create_qrel_results(3, 3)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for i, query in enumerate(qrel):
        p = ctx.Process(
            target=run_eval,
            kwargs={
                "rank": i,
                "qrel": {query: qrel[query]},
                "result": {query: result[query]},
                "output_queue": q,
            },
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    res1 = {}
    while q.qsize() > 0:
        score = q.get()
        res1.update(score)

    res2 = run_eval(0, qrel, result)
    assert res1 == res2, (res1, res2)
    return


def create_query_groups(qrel, n_groups):
    qids = list(qrel.keys())
    gsz = [len(qids) // n_groups] * n_groups
    gsz[0] += len(qrel) % n_groups

    i = 0
    q_groups = []
    for sz in gsz:
        q_groups.append(qids[i : i + sz])
        i += sz
    return q_groups


def main(num_queries: int, num_docs: int, num_queries_per_proc: int = 1, test=False):
    test = bool(test)
    num_queries = int(num_queries)
    num_docs = int(num_docs)
    num_queries_per_proc = int(num_queries_per_proc)

    qrel, result = create_qrel_results(num_queries, num_docs)
    qrel_keys = list(qrel.keys())
    num_procs = psutil.cpu_count()
    # query_groups = create_query_groups(qrel, n_groups= psutil.cpu_count())
    # print("number of query groups: %s" % len(query_groups))

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    start = time.time()
    j = 0
    res1 = {}
    while j < len(qrel_keys):
        procs = []
        for i in range(num_procs):
            if j >= len(qrel_keys):
                break
            _qrel = {k: qrel[k] for k in qrel_keys[j : j + num_queries_per_proc]}
            _result = {k: result[k] for k in qrel_keys[j : j + num_queries_per_proc]}
            if test == False:
                for k in qrel_keys[j : j + num_queries_per_proc]:
                    del qrel[k]
                    del result[k]
            j += num_queries_per_proc
            p = ctx.Process(
                target=run_eval,
                kwargs={"rank": i, "qrel": _qrel, "result": _result, "output_queue": q},
            )
            p.start()
            procs.append(p)

        print("All subprocesses spawed")
        for p in procs:
            p.join()

    while q.qsize() > 0:
        score = q.get()
        res1.update(score)
    end = time.time()

    print("Total time for evaluation: %s" % (end - start))

    if test:
        start = time.time()
        res2 = run_eval(0, qrel, result)
        assert res1 == res2, (res1, res2)
        end = time.time()
        print("Total time for a single process evalution: %s" % (end - start))

    return res1


if __name__ == "__main__":
    import fire
    import sys

    main(*sys.argv[1:])
