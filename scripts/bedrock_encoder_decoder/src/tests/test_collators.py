import hydra
import collators
import numpy as np

def test_concat_batch_util_size():
    """
    Tests that concatenating batches produces the expected size
    """

    with hydra.initialize("../config/tests"):
        cfg = hydra.compose(config_name="collator_utils")

    tokenizer = hydra.utils.call(cfg.tokenizer)

    batches = [hydra.utils.instantiate(cfg.batches[key]) for key in cfg.batches]

    concatenated_batches = collators.utils.pad_and_concatenate_batches(batches=batches, pad_token_id=tokenizer.pad_token_id)

    for key in cfg.expected_sizes:
        got_shape = concatenated_batches[key].shape
        expected_shape = cfg.expected_sizes[key]
        assert expected_shape == got_shape, f"Expected shape {expected_shape} got shape {got_shape}"

def test_concat_single_batch():
    """
    Tests that passing a list with only one batch works
    """

    with hydra.initialize("../config/tests"):
        cfg = hydra.compose(config_name="collator_utils")

    # batches is actually just 1 element
    batches = [hydra.utils.instantiate(cfg.batches[key]) for key in cfg.batches][:1]
    
    tokenizer = hydra.utils.call(cfg.tokenizer)

    concatenated_batch = collators.utils.pad_and_concatenate_batches(batches=batches, pad_token_id=tokenizer.pad_token_id)
    
    for key in batches[0]:
        assert np.allclose(batches[0][key],concatenated_batch[key]), "Concatenating only one batch, input and output are the same failed on key {key}"

def test_concat_zero_batch():
    """
    Tests that passing an empty list raises a ValueError list with only one batch works
    """

    with hydra.initialize("../config/tests"):
        cfg = hydra.compose(config_name="collator_utils")

    # batches is actually just 1 element
    batches = []
    
    tokenizer = hydra.utils.call(cfg.tokenizer)
    try:
        concatenated_batch = collators.utils.pad_and_concatenate_batches(batches=batches, pad_token_id=tokenizer.pad_token_id)
    except ValueError as e:
        return

    assert False, "Passing an empty list should have returned a ValueError"
