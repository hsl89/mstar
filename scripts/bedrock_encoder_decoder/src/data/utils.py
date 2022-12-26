import datasets
# don't rely on cached index files
datasets.disable_caching()
import logging

def construct_hf_dataset_from_filepaths(paths: list) -> datasets.arrow_dataset.Dataset:
    """Given a list of paths to .arrow objects, construct a dataset by concatenation"""
    arrow_datasets = [datasets.arrow_dataset.Dataset.from_file(x) for x in paths]
    logging.warning(
        "Concatenating arrow datasets can slow down .take() "
        "Make sure to use pre-fetch+multiprocessing."
    )
    concatenated_dataset = datasets.concatenate_datasets(arrow_datasets)
    return concatenated_dataset
