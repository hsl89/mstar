import fileinput
from transformers import AutoTokenizer

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('facebook/mbart-large-cc25')
    num_tokens = 0
    lines = []
    for l in fileinput.input():
        lines.append(l.strip())
        if len(lines) >= 10000:
            toks = tokenizer._tokenizer.encode_batch(lines, add_special_tokens=False,
                                                     is_pretokenized=False)
            num_tokens += sum(len(t) for t in toks)
            lines = []
    if len(lines):
        toks = tokenizer._tokenizer.encode_batch(lines, add_special_tokens=False,
                                                 is_pretokenized=False)
        num_tokens += sum(len(t) for t in toks)
    print(num_tokens)
