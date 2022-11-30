from .perturber import InputPerturber
import unicodedata
import random
import string
from .utils import Tokenizer
from english_words import english_words_alpha_set, web2_words_set
ENGLISH_WORD_SET = english_words_alpha_set | web2_words_set
STRING_PRINTABLE_SET = set(string.printable)

def _per_word_perturb_limit(word):
    return min(3, len(word) // 2)

FUZZ_MAP = {'q': 'wasedzx', 
            'w': 'qesadrf', 
            'e': 'wrsfdqagt', 
            'r': 'etdgfwsgt', 
            't': 'ryfhgedju', 
            'y': 'tugjhrfji', 
            'u': 'yihkjtglo', 
            'i': 'uojlkyhlp', 
            'o': 'ipkluj', 
            'p': "loik", 
            'a': 'qszwxdce', 
            's': 'wxadrfv', 
            'd': 'ecsfaqgbv', 
            'f': 'dgrvwsxyhn', 
            'g': 'tbfhedcyjn', 
            'h': 'yngjfrvkim', 
            'j': 'hknugtblom', 
            'k': 'jlinyhn', 
            'l': 'okmpujn', 
            'z': 'axsvd', 
            'x': 'zcsdbvf', 
            'c': 'xvdfzsgb', 
            'v': 'cfbgxdn', 
            'b': 'vnghcfn', 
            'n': 'bmhjvgk', 
            'm': 'nkjlk',
            ' ': 'xcvbnm,'
            }

def _get_random_letter(letter):
    normalized_letter = normalize(letter).lower()
    if normalized_letter not in FUZZ_MAP.keys():
        return letter
    chosen_letter = random.choice(FUZZ_MAP[normalized_letter])
    chosen_letter = preserve_new_letter_case(letter, chosen_letter)
    return chosen_letter

def normalize(letter):
    return unicodedata.normalize('NFKD', letter).encode('ascii', 'ignore').decode('ascii')

def preserve_new_letter_case(old_letter, new_letter):
    if (set(normalize(old_letter)) - STRING_PRINTABLE_SET) \
     or (set(normalize(new_letter)) - STRING_PRINTABLE_SET):
        print(old_letter, normalize(old_letter), new_letter, normalize(new_letter))
    assert not (set(normalize(old_letter)) - STRING_PRINTABLE_SET) and not (set(normalize(old_letter)) - STRING_PRINTABLE_SET)
    if old_letter.isupper():
        return new_letter.upper()
    elif old_letter.islower():
        return new_letter.lower()
    else:
        return new_letter

class AddLetterPerturber(InputPerturber):
    """
    Add a random neighboring letter on the keyboard while ensuring the resulting word is not a another English word.
    """
    def __init__(self, seed: int, perturb_prob: float) -> None:
        super().__init__(seed, perturb_prob)
        self.tokenizer = Tokenizer()

    def _add_letters(self, word):
        candidates = [word]

        for _ in range(_per_word_perturb_limit(word)):
            candidate = list(random.choice(candidates))
            for i in range(len(candidate)+1):
                current_char = candidate[min(i, len(candidate)-1)] # Handle index out of bounds case for last iteration
                new_word = ''.join(candidate[:i] + [_get_random_letter(current_char)] + candidate[i:])
                if new_word.lower() not in ENGLISH_WORD_SET:
                    candidates.append(new_word)
        
        candidates = candidates[1:]
        
        if candidates:
            return random.choice(candidates) 
        else:
            return word

    def perturb(self, text: str) -> str:
        random.seed(self.seed)
        tokens = self.tokenizer.tokenize(text)

        for i, token in enumerate(tokens):
            if token.isalpha() and len(token) > 1 and random.random() < self.perturb_prob:
                tokens[i] = self._add_letters(token)
        
        return self.tokenizer.detokenize(tokens)

class AddWhitespaceBeforePunctuationPerturber(InputPerturber):
    """
    Insert whitespace characters before punctuation characters.
    """
    def perturb(self, text: str) -> str:
        random.seed(self.seed)
        result = []
        for char in text:
            if char in string.punctuation and random.random() < self.perturb_prob:
                result.append(' ')
            result.append(char)
        return ''.join(result)

class DropPunctuationPerturber(InputPerturber):
    """
    Drops punctuation characters.
    """
    def perturb(self, text: str) -> str:
        random.seed(self.seed)
        result = []
        for char in text:
            if char not in string.punctuation or random.random() >= self.perturb_prob:
                result.append(char)
        return ''.join(result)

class DropLetterPerturber(InputPerturber):
    """
    Drops letters while ensuring the resulting word is not a another English word
    """
    def __init__(self, seed: int, perturb_prob: float) -> None:
        super().__init__(seed, perturb_prob)
        self.tokenizer = Tokenizer()

    def _drop_letter(self, word):
        candidates = [word]
        for _ in range(_per_word_perturb_limit(word)):
            candidate = list(random.choice(candidates))
            for i in range(len(candidate)):
                new_word = ''.join(candidate[:i] + candidate[i+1:])
                if new_word.lower() not in ENGLISH_WORD_SET:
                    candidates.append(new_word)
        
        candidates = candidates[1:]
        if candidates:
            return random.choice(candidates) 
        else:
            return word

    def perturb(self, text: str) -> str:
        random.seed(self.seed)
        tokens = self.tokenizer.tokenize(text)

        for i, token in enumerate(tokens):
            if token.isalpha() and len(token) > 1 and random.random() < self.perturb_prob:
                tokens[i] = self._drop_letter(token)
        
        return self.tokenizer.detokenize(tokens)


class DropWhitespacePerturber(InputPerturber):
    """
    Drops whitespace characters.
    """
    def perturb(self, text: str) -> str:
        random.seed(self.seed)
        result = []
        for char in text:
            if char not in string.whitespace or random.random() >= self.perturb_prob:
                result.append(char)
        return ''.join(result)

class DropWhitespaceAroundPunctuationPerturber(InputPerturber):
    """
    Drops whitespace characters next to punctuation characters.
    """
    def perturb(self, text: str) -> str:
        random.seed(self.seed)
        result = []
        i = 0

        while i < len(text) - 1:
            if text[i] in string.whitespace and text[i+1] in string.punctuation and random.random() < self.perturb_prob:
                i += 1
                continue
            
            result.append(text[i])

            if text[i] in string.punctuation and text[i+1] in string.whitespace and random.random() < self.perturb_prob:
                i += 1
         
            i += 1
        
        if i == len(text) - 1:
            result.append(text[-1])
        
        return ''.join(result)

class LetterCasePerturber(InputPerturber):
    """
    Based on https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/change_char_case/transformation.py
    """
    def perturb(self, text: str) -> str:
        random.seed(self.seed)
        result = []
        for char in text:
            if random.random() < self.perturb_prob:
                char = char.swapcase()
            result.append(char)
        return ''.join(result)


class ReplaceWithRandomCharacterPerturber(InputPerturber):
    """
    Based on https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/change_char_case/transformation.py
    """

    def perturb(self, text: str) -> str:
        random.seed(self.seed)
        result = []
        for char in text:
            if random.random() < self.perturb_prob:
                char = random.choice(string.printable)
            result.append(char)
        return ''.join(result)

class RepeatLetterPerturber(InputPerturber):
    """
    Repeat letters a random number of times while ensuring the resulting word is not a another English word.
    """
    def __init__(self, seed: int, perturb_prob: float) -> None:
        super().__init__(seed, perturb_prob)
        self.tokenizer = Tokenizer()

    def _repeat_letters(self, word):
        chars = list(word)
        result = []
        perturb_limit = _per_word_perturb_limit(word)
        perturb_count = 0

        # Limit number of perturbations to 1
        for i in range(len(chars)):
            if perturb_count < perturb_limit and random.random() < self.perturb_prob:
                repeated_chars = [chars[i]] * random.randrange(1, 3)
                if ''.join(result + repeated_chars + chars[i:]).lower() not in ENGLISH_WORD_SET:
                    result += [chars[i]] * random.randrange(1, 3)
                    perturb_count += 1
            result.append(chars[i])
        return ''.join(result)
    
    def perturb(self, text: str) -> str:
        random.seed(self.seed)
        tokens = self.tokenizer.tokenize(text)

        for i, token in enumerate(tokens):
            if token.isalpha() and len(token) > 1:
                tokens[i] = self._repeat_letters(token)
        
        return self.tokenizer.detokenize(tokens)

class RepeatPunctuationPerturber(InputPerturber):
    """
    Repeat punctuation characters a random number of times.
    """
    def perturb(self, text: str) -> str:
        random.seed(self.seed)
        result = []
        for char in text:
            if char in string.punctuation and random.random() < self.perturb_prob:
                result += [char] * random.randrange(1, 3)
            result.append(char)
        return ''.join(result)

class RepeatWhitespacePerturber(InputPerturber):
    """
    Repeats whitespace characters a random number of times.
    """
    def perturb(self, text: str) -> str:
        random.seed(self.seed)
        result = []
        for char in text:
            if char in string.whitespace and random.random() < self.perturb_prob:
                result += [char] * random.randrange(1, 5)
            result.append(char)
        return ''.join(result)

class SubstituteLetterPerturber(InputPerturber):
    """
    Swap a letter with another random neighboring letter on the keyboard while ensuring the resulting word is not a another English word.
    """
    def __init__(self, seed: int, perturb_prob: float) -> None:
        super().__init__(seed, perturb_prob)
        self.tokenizer = Tokenizer()

    def _sub_letters(self, word):
        candidates = [word]

        for _ in range(_per_word_perturb_limit(word)):
            candidate = list(random.choice(candidates))
            for i in range(len(candidate)):
                current_char = candidate[min(i, len(candidate)-1)] # Handle index out of bounds case for last iteration
                new_word = ''.join(candidate[:i] + [_get_random_letter(current_char)] + candidate[i+1:])
                if new_word.lower() not in ENGLISH_WORD_SET:
                    candidates.append(new_word)
        
        candidates = candidates[1:]
        
        if candidates:
            return random.choice(candidates) 
        else:
            return word

    def perturb(self, text: str) -> str:
        random.seed(self.seed)
        tokens = self.tokenizer.tokenize(text)

        for i, token in enumerate(tokens):
            if token.isalpha() and len(token) > 1 and random.random() < self.perturb_prob:
                tokens[i] = self._sub_letters(token)
        
        return self.tokenizer.detokenize(tokens)

class SwapAdjacentLetterPerturber(InputPerturber):
    """
    Swap a letter with an adjacent letter on the keyboard while ensuring the resulting word is not a another English word.
    """
    def __init__(self, seed: int, perturb_prob: float) -> None:
        super().__init__(seed, perturb_prob)
        self.tokenizer = Tokenizer()

    def swap(self, word, i, j):
        chars = list(word)
        tmp = preserve_new_letter_case(chars[j], chars[i])
        chars[i] = preserve_new_letter_case(chars[i], chars[j])
        chars[j] = tmp
        return ''.join(chars)

    def _get_swap_candidate_ids(self, i, list_len):
        if i == 0:
            return [(0, 1)]
        elif i == list_len - 1:
            return [(-1, -2)]
        else:
            return [(i-1, i), (i, i+1)]

    def _perturb_word(self, word):    
        if len(word) < 2:
            return word
        if len(word) == 2:
            return self.swap(word, 0, 1)
        if len(word) == 3:
            if random.random() < 0.5:
                return self.swap(word, 0, 1)
            else:
                return self.swap(word, 1, 2)


        candidates = []

        for i in range(len(word)):
            ids_to_swap = random.choice(self._get_swap_candidate_ids(i, len(word)))
            new_word = self.swap(word, *ids_to_swap)
            if new_word.lower() not in ENGLISH_WORD_SET:
                candidates.append(new_word)
        
        if candidates:
            return random.choice(candidates) 
        else:
            return word

    def perturb(self, text: str) -> str:
        random.seed(self.seed)
        tokens = self.tokenizer.tokenize(text)

        for i, token in enumerate(tokens):
            if token.isalpha() and len(token) > 1 and random.random() < self.perturb_prob:
                tokens[i] = self._perturb_word(token)
        
        return self.tokenizer.detokenize(tokens)