export PYTHONUNBUFFERED=1 NCCL_DEBUG=WARN TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=16

python3 main.py model=distilbert task=sts_tests use_fake_inference=True
# python3 main.py model=sgpt_125m_nli task=sts_tests use_fake_inference=True

python3 main.py model=distilbert task.task_type=retrieval task=retrieval_tests use_fake_inference=True
# python3 main.py model=sgpt_125m_nli task=retrieval_tests use_fake_inference=True

python3 main.py model=distilbert task=clustering_tests use_fake_inference=True
# python3 main.py model=sgpt_125m_nli task=clustering_tests use_fake_inference=True
