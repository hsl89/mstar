TOKENIZERS_PARALLELISM=false OPENBLAS_NUM_THREADS=16 python3 main.py model=sup_simcse model.data_params.batch_size=64 task.task_type=clustering task.task_name=[ArxivClusteringP2P]
