import mstar.AutoTokenizer
import datasets
import numpy as np
import queue
import torch
import hydra
import logging
import utils

# the huggingface column name we will take the text from
DATASET_KEY = "text"

# don't rely on cached index files
# datasets shuffle will try to cache index files
datasets.disable_caching()

#TODO: current throws away >100K characters
# Otherwise tokenizing long strings can delay training
MAX_CHAR_LENGTH=50000

class OnlinePackedDataset(torch.utils.data.IterableDataset):
    """
    Takes a HF arrow dataset. 
    Assumes it is already sharded.
    Performs online packing.
    Finally, detokenizes data to pass to the collator
    """

    def __init__(
        self,
        hf_dataset: datasets.arrow_dataset.Dataset,
        tokenizer,
        max_tokens_per_example: int, #pack to this length
        base_seed: int,
        data_collator,
        partition: str,  # train/test/val
        process_global_rank: int,
        detokenize: bool,  # whether to detokenize back to text
    ):

        # TODO: tokenizer max length 50k for long string truncation?

        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_tokens_per_example = max_tokens_per_example

        self.data_collator = data_collator
        assert partition in ["train","val","test"]
        self.partition = partition
        self.base_seed = base_seed
        # will use process rank to print info messages
        # also useful for load/save
        self.process_global_rank = process_global_rank
        
        self.detokenize = detokenize
        self.token_buffer = []
        self.current_epoch = 0

        # we will use this to track which example 
        # we are on in the hf dataset
        self.current_example_index = 0

        self._reshuffle_dataset()
        
    def _end_of_pass_reset(self):
        """
        Reshuffle and reset at the end of one pass through the data.
        """
        # increment current epoch
        # dataset reshuffle relies on this increment
        self.current_epoch += 1
        self._reshuffle_dataset()
        self.current_example_index = 0

    def _reshuffle_dataset(self):
        """
        Shuffle underlying HF dataset. Used at the end of pass 
        through the dataset to avoid the same data order.
        """
        if self.partition == 'train':
            shuffle_seed = self.base_seed + self.current_epoch
        else:
            assert self.partition in ['val','test']
            shuffle_seed = self.base_seed

        logging.info(
            f"Reshuffling dataset on worker {self.process_global_rank}"
        )
        self.shuffled_hf_dataset = self.hf_dataset.shuffle(seed=shuffle_seed)

    def state_dict(self) -> dict:
        """
        Return dict containing all info needed for deterministic resume
        """
        state_dict = {
            "token_buffer": self.token_buffer,
            "current_example_index": self.current_example_index,
            "current_epoch": self.current_epoch,
        }
        return state_dict

    def load_state_dict(self, state_dict: dict):
        """
        Load all info needed for deterministic resume
        """
        self.token_buffer = state_dict["token_buffer"]
        self.current_example_index = state_dict["current_example_index"]
        self.current_epoch = state_dict["current_epoch"]

        #reshuffle required to ensure that the dataset 
        #has been shuffled with the same seed as before
        self._reshuffle_dataset()

    @property
    def token_buffer_length(self) -> int:
        return len(self.token_buffer)


    def _read_example(self):
        """
        Reads a text example from a dataset
        """
        # reset and reshuffle if we have completed a full pass
        if self.current_example_index>=len(self.hf_dataset)-1:
            # assumes we have completed an epoch
            self._end_of_pass_reset()
        
        # get next example
        example = self.shuffled_hf_dataset[self.current_example_index][DATASET_KEY]

        #avoid blocking on tokenizing very long sequences
        #TODO: maintained a queue of examples, and don't discard the end of the example
        if len(example)>MAX_CHAR_LENGTH:
            pointer = MAX_CHAR_LENGTH
            #find the first space, then truncate
            while example[pointer]!=' ' and pointer>=0:
                pointer-=1
               
            if pointer==0:
                #no spaces in doc?
                example = example[:MAX_CHAR_LENGTH]
            else:
                #truncate at space character
                example = example[:pointer] 
                 
        # increment index for next example
        self.current_example_index+=1 

        return example

    def _append_to_token_buffer(self, tokenized_example: list):
        # TODO: extend to attention mask
        # for block-diagonal, requires tracking example boundaries
        self.token_buffer.extend(tokenized_example["input_ids"])

    def _read_tokens_from_token_buffer(self, num_tokens: int):

        # if buffer is empty, read that many tokens in
        while self.token_buffer_length < num_tokens:
            # get un-tokenized example
            example = self._read_example()
            tokenized_example = self.tokenizer(example)
            # add tokenized example to buffer
            self._append_to_token_buffer(tokenized_example)

        to_return = self.token_buffer[:num_tokens]
        # remove used tokens from buffer
        # better to rely on list for a single array-backed op
        # is there a better option
        self.token_buffer = self.token_buffer[num_tokens:]

        return to_return

    def _process_example(self, example_tokens: list) -> list:
        """
        Wrapper method to convert raw tokens to a batch
        batch_tokens: list of list of ints
        """

        # TODO:trim if necessary
        # TODO: append EOS if necessary, or remove EOS from end

        if self.detokenize:
            # TODO: temporary hack to avoid intergrating collator
            return {'text':self.tokenizer.decode(example_tokens)}
        else:
            return example_tokens

    def __next__(self):

        example_tokens = self._read_tokens_from_token_buffer(self.max_tokens_per_example)

        processed_example = self._process_example(example_tokens)

        return processed_example

    def __iter__(self):
        return self
