import unittest
from mstar.tokenizers.atm_tokenizers import ATMTokenizerFast
from mstar.utils.hf_utils import get_tokenizer_file_from_s3

class ATMTokenizerFastTest(unittest.TestCase):
    def test_tokenizer(self):
        cache_dir = "/tmp"
        key = "atm-PreLNSeq2Seq-20B"
        revision = "main"
        downloaded_folder = get_tokenizer_file_from_s3(
            key, revision=revision, cache_dir=cache_dir
        )

        tokenizer = ATMTokenizerFast(vocab_file=f"{downloaded_folder}/spiece.model")

        test = "this is fake data"
        encoded = tokenizer(test, return_tensors="pt")
        decoded = tokenizer.batch_decode(encoded['input_ids'], skip_special_tokens=True)

        expected_ids = [[1, 2156, 2060, 27497, 3133, 2]]

        self.assertEqual(encoded['input_ids'].tolist(), expected_ids)
        self.assertEqual(decoded[0], "<s> this is fake data</s>")
