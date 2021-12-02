import mstar
from mstar.uf_format.uf_reader import SeqTagTask, SeqTagDataReader
import json
import argparse

def test_uf_reader():
    """
    Reads in file, creates uf_reader object and outputs tokens and tags
    """
    f = open("tests/test_ip.json")
    reader = SeqTagDataReader((SeqTagTask.NER,))
    for line in f:
        ip = json.loads(line)
        d = reader.read_json(ip)
        # print(d)
        # print(d.tokens)
        assert len(d.tokens) > 0
        for tok in d.tokens:
           # print(tok.text, tok.attributes['tag'])
           assert type(tok.text) == str
           assert tok.attributes['tag'] is not None
