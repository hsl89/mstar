"""
Test online example packing dataset
"""

import time
import torch
import hydra
import numpy as np
from data.utils import construct_hf_dataset_from_filepaths


def test_iteration_speed():
    """
    Test to ensure that online packing is fast enough
    """
    PER_STEP_TIME_THRESHOLD = 0.1

    with hydra.initialize("../config/tests"):
        cfg = hydra.compose(config_name="online_packing")

    # get data from s3 if not already there
    hydra.utils.call(cfg.download_data_fn)

    # should not test until after saving
    assert cfg.load_save.test_iter > cfg.load_save.save_iter
    hf_dataset = construct_hf_dataset_from_filepaths(cfg.training_datasets).shard(1000,1)

    tokenizer = hydra.utils.call(cfg.tokenizer)

    dataset = hydra.utils.instantiate(
        cfg.online_packed_dataset, hf_dataset=hf_dataset, tokenizer=tokenizer
    )

    times = []
    start_time = time.monotonic()
    for i, _ in enumerate(dataset):
        end_time = time.monotonic()
        times.append(end_time-start_time)
        start_time=end_time 
        if i > cfg.max_steps:
            break
    
    max_step_time = max(times)
    assert (
        max_step_time < PER_STEP_TIME_THRESHOLD
    ), f"Max step time of {max_step_time:.2f} does not meet {PER_STEP_TIME_THRESHOLD:.2f} target"


def test_dataset_save_load():
    """
    Test to ensure that single worker dataset save/load captures full state
    Tests equivalence after save/load and ensures this test covers more than 1 
    pass through the data
    """
    with hydra.initialize("../config/tests"):
        cfg = hydra.compose(config_name="online_packing")

    # get data from s3 if not already there
    hydra.utils.call(cfg.download_data_fn)
    # should not test until after saving
    assert cfg.load_save.test_iter > cfg.load_save.save_iter

    hf_dataset = construct_hf_dataset_from_filepaths(cfg.training_datasets)

    #truncate via sharding to make it faster to run over an epoch
    hf_dataset = hf_dataset.shard(cfg.sharding.num_shards, cfg.sharding.index)

    tokenizer = hydra.utils.call(cfg.tokenizer)

    dataset = hydra.utils.instantiate(
        cfg.online_packed_dataset, hf_dataset=hf_dataset, tokenizer=tokenizer
    )

    first_run_examples = []
    for i, example in enumerate(dataset):
        if i == cfg.load_save.save_iter:
            state_dict = dataset.state_dict()
            torch.save(state_dict, cfg.load_save.state_dict_path)
            save_epoch = dataset.current_epoch

        if i > cfg.load_save.save_iter:
            first_run_examples.append((example,dataset.current_example_index,dataset.current_epoch))

        if i== cfg.load_save.test_iter:
            break

    assert first_run_examples, "Make sure we get a nonzero number of test examples"

    second_run_examples = []
    dataset.load_state_dict(torch.load(cfg.load_save.state_dict_path))

    for i, example in enumerate(dataset):
        second_run_examples.append((example,dataset.current_example_index,dataset.current_epoch))
        if i == cfg.load_save.test_iter - cfg.load_save.save_iter - 1:
            break

    test_epoch = dataset.current_epoch

    assert test_epoch > save_epoch, "To ensure this test covers more than 1 epoch, need to increase gap between save iter and load iter"

    for i,(first_run_example, second_run_example) in enumerate(zip(first_run_examples, second_run_examples)):
        assert first_run_example==second_run_example
