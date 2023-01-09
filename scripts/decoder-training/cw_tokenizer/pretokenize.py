"""
Collection of tools for pretokenization
"""
import regex as re


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.
    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    greek = ['Γ', 'Δ', 'Θ', 'Λ', 'Ξ', 'Π', 'Σ', 'Φ', 'Ψ', 'Ω'] \
            + [chr(c) for c in range(ord('α'), ord('λ'))] \
            + [chr(c) for c in range(ord('π'), ord('ω'))]
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1

    cs = [chr(n) for n in cs]
    d = dict(zip(bs, cs))
    exceptions = [144, 145, 157, 158, 168, 170, 175, 178, 179, 180, 181, 184, 185, 186, 188, 189, 190]
    for i in range(len(exceptions)):
        d[exceptions[i]] = greek[i]
    return d


def convert_unicode(text):
    encoded_text = "".join(
        [BYTE_ENCODER[c] for c in text.encode("utf-8", errors="replace")]
    )
    return encoded_text


def pretokenize(text, pattern):
    pretokenized = []
    for token in re.findall(pattern, text):
        encoded_token = "".join(
            [BYTE_ENCODER[c] for c in token.encode("utf-8", errors="replace")]
        )
        pretokenized.append(encoded_token)

    pretokenized_text = " ".join(pretokenized)
    pretokenized_text = pretokenized_text.replace("< unk >", "<unk>")
    pretokenized_text = pretokenized_text.replace("< s >", "<s>")
    pretokenized_text = pretokenized_text.replace("</ s >", "</s>")
    pretokenized_text = pretokenized_text.replace("< en _ XX >", "<en_XX>")
    pretokenized_text = pretokenized_text.replace("< prog _ PY >", "<prog_PY>")
    pretokenized_text = pretokenized_text.replace("< prog _ JAVA >", "<prog_JAVA>")
    pretokenized_text = pretokenized_text.replace("< prog _ JS >", "<prog_JS>")
    pretokenized_text = pretokenized_text.replace("< prog _ CS >", "<prog_CS>")
    pretokenized_text = pretokenized_text.replace("< prog _ TS >", "<prog_TS>")
    return pretokenized_text


def reverse_pretokenize(tokens):
    if not isinstance(tokens, str):
        tokens = "".join(tokens)  # convert iterable to string

    tokens = tokens.replace(" ", "")
    text = bytearray([BYTE_DECODER[c] for c in tokens]).decode(
        "utf-8", errors="replace"
    )
    return text


# GLOBALS
BYTE_ENCODER = bytes_to_unicode()
BYTE_DECODER = {v: k for k, v in BYTE_ENCODER.items()}

# Pretokenize 0: the original GPT-2 style is:
# i.e. `    print something` -> [`   `, ` print`, ` something`]
gpt2_pretokenize_pattern = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\n+|\t+|\s+(?!\S)| +|\s+"""
)
# pretokenization 1: will not include leading spaces with words/numbers/etc
# i.e. `    print something` -> [`    `, `print`, ` `, `something`]
no_space_prefix_pattern = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+|\n+|\t+| +|\s+"""
)

# pretokenization 2: similar to GPT-2, pretokenize but do *not*
# break up indentation in order to prefix tokens with the space
# i.e. `    print something` -> [`    `, `print`, ` something`]
space_prefix_pattern = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d|^ {2,}| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\n+|\t+| +|\s+|\s+(?!\S)"""
)

newline_pattern = re.compile(
    r"""\n+|[^\n]+"""
)

pretokenize_regex = gpt2_pretokenize_pattern