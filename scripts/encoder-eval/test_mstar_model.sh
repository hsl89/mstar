# python3 main.py model=distilbert task=sts_tests
# python3 main.py model=sgpt_125m_nli task=sts_tests

python3 main.py model=mstar_5b task.task_type=retrieval task.task_name=[SCIDOCS,SciFact]
# python3 main.py model=sgpt_125m_nli task=retrieval_tests

# python3 main.py model=distilbert task=clustering_tests
# python3 main.py model=sgpt_125m_nli task=clustering_tests
