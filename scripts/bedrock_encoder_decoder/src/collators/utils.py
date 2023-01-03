import hydra
import numpy as np
from typing import List, Dict
import logging


def validate_batch_keys(batches: List[dict]) -> List[str]:
    """
    Checks that all elements of the list of batches are
    dictionaries that have the same keys. This allows them 
    to be easily concatenated by key
    Args:
        batches: a list of dictionaries - each element is a batch, structured
                    as a dictionary. Keys are strings and values are numpy arrays.
    Returns:
        expected_keys: a list of strings - describes the array elements of each batch
    """
    expected_keys = set(batches[0].keys())
    for i,batch in enumerate(batches[1:]):
        got_keys = set(batch.keys())
        assert expected_keys == got_keys, f"Expected {expected_keys} got {got_keys}"

    return expected_keys

def validate_shapes_and_return_max(batches: List[dict]) -> Dict[str,int]:
    """
    Checks that the items in each batch (element of batches) can be
    stacked along the first dimension after padding the last dimension
    Args:
        batches: a list of dictionaries - each element is a batch, structured
                    as a dictionary. Keys are strings and values are numpy arrays.
    Returns:
        max_final_dim_shapes: a dictionary - each value lists the final padding 
                                dimension required for that key
    """
    first_batch_truncated_shape = {key: list(val.shape)[1:-1] for key, val in batches[0].items()}
    max_final_dim_shapes = {key: list(val.shape)[-1] for key, val in batches[0].items()}

    for batch in batches[1:]:
        for key, val in batch.items():
            #check item dimensions match first batch item dimensions
            truncated_shape = list(val.shape)[1:-1] 
            assert truncated_shape==first_batch_truncated_shape[key], f"Batch shape {truncated_shape} does not match first batch shape {first_batch_truncated_shape[key]}" 
            #update max size
            final_dim_size = list(val.shape)[-1]
            
            max_final_dim_shapes[key] = max(max_final_dim_shapes[key],final_dim_size)

    return max_final_dim_shapes


def pad_tensor_to_shape(to_pad, final_dim_shape:int, pad_token_id: int):
    """
    Pads tensor along the final dimension so a batch can be formed 
    by concatenation. Currently assumes every tensor is 2-D.
    Args:
        to_pad: a numpy array - array that will be padded
        final_dim_shape: an integer - the size up to which we will pad the final dimension of to_pad 
        pad_token_id: an integer - we use the pad_token_id for padding
    Returns:
        padded_tensor: a numpy array - to_pad, padded up to final_dim_shape, using pad_token_id
    """
    assert len(to_pad.shape)==2, "Only supports padding for 2-D, got {len(to_pad.shape)}-D"

    padded_tensor = np.pad(
                        to_pad, 
                        [(0,0),(0,final_dim_shape-to_pad.shape[-1])], 
                        mode="constant", 
                        constant_values = pad_token_id
                )

    return padded_tensor

def pad_batch_to_shapes(batch: Dict[str,np.ndarray], padded_shapes: Dict[str,int], pad_token_id: int, keys: List[str]) -> dict:
    """
    Right pads all batch items based on the input shapes.
    Args:
        batch: a dict - keys describe elements, values are numpy arrays to pad
        padded_shapes: a dict - maps a batch element description to its padding shape for the final dim 
        pad_token_id: an integer - we use the pad_token_id for padding
    Returns:
        padded_tensor: a numpy array - to_pad, padded up to final_dim_shape, using pad_token_id
    """
    padded_batch = {key: pad_tensor_to_shape(val, padded_shapes[key], pad_token_id) for key,val in batch.items()}

    return padded_batch
    

def pad_and_concatenate_batches(batches: List[dict], pad_token_id: int) -> Dict[str,np.ndarray]:
    """
    Takes a list of dicts, where each dict is a batch. Then checks that each dict has 
    the same keys to ensure we are not losing info by concatenation. Then pads all 
    tensors along the final dimension so that they can be concatenated key-wise .
    Args:
        batches: a list of dictionaries - each element is a batch, structured
                    as a dictionary. All batches will be padded.
        pad_token_id: an integer - we use the pad_token_id for padding
    Returns:
        final_batch: a dict - batch represented as a dict, where each batch has in batches
                                has been concatenated key-wise
    """
    if len(batches)==0:
        raise ValueError("Concatenating batches but got length 0 input ")

    elif len(batches)==1:
        #if there's only one batch, return it
        logging.warning("Concatenating batches but only got one batch")
        return batches[0]

    # make sure all batches have the same keys  
    keys = validate_batch_keys(batches)

    # pad all batch keys to the same shapes
    shapes = validate_shapes_and_return_max(batches)
    padded_batches = [pad_batch_to_shapes(batch=batch, padded_shapes=shapes, pad_token_id=pad_token_id, keys=keys) for batch in batches]

    #concatenate key-wise after padding 
    final_batch = {
            key: np.concatenate([padded_batch[key] for padded_batch in padded_batches])
            for key in keys
        }

    return final_batch
