from mteb.tasks import HotpotQA, SCIDOCS
import random
from time import time
import tracemalloc

"""
see mteb.tasks.AbsTaskRetrieval for steps of evaluating a model
on an retrieval task
"""


def retrieve(corpus, queries):
    """mimic retrieve step to produce a dictionary of relevance score"""
    s = {cid: random.uniform(-1.0, 1.0) for cid in corpus}
    res = {qid: s for qid in queries}
    return res


def evaluate(relevant_docs, results, k_values):
    from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
    from beir.retrieval.evaluation import EvaluateRetrieval

    # if only use 1 k value, then eval should not encounter oom issue
    retriever = EvaluateRetrieval(None, score_function="cos_sim", k_values=k_values)
    print("k values: ", retriever.k_values)
    retriever.evaluate(relevant_docs, results, retriever.k_values)
    return


def run():
    t = HotpotQA()
    # t = SCIDOCS()
    t.load_data()

    start_time = time()
    results = retrieve(t.corpus["test"], t.queries["test"])
    end_time = time()
    k = list(results.keys())[0]
    v = results[k]
    print(
        "results size:%s\nNumber of queries:%s\nNumber of corpus:%s\n"
        % (len(results), len(results), len(v))
    )
    print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
    evaluate(t.relevant_docs["test"], results, k_values=[10])
    return


if __name__ == "__main__":
    tracemalloc.start()
    run()
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)
