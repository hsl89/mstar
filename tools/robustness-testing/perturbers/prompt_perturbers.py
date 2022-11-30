# TODO: Expand example and shot-separators to include the ones used in Bedrock:https://quip-amazon.com/znNVAzRlBvH4/PC-MTL-pre-fine-tuning-for-Bedrock#temp:C:VJL8a916b5b79e341cc808fdecc1
from abc import abstractmethod
from .perturber import PromptPerturber
from typing import List
import random
import string

class ExampleOrderPerturber(PromptPerturber):
    def perturb(self, text: str) -> str:
        return text

    def perturb_order(self, indices: List[int]) -> str:
        random.seed(self.seed)
        return random.shuffle(indices)

class ExampleSeparatorPerturber(PromptPerturber):
    def perturb(self, text: str) -> str:
        return text
    
    @abstractmethod
    def get_example_separator(self) -> str:
        pass

class ExampleSeparatorNewlinePerturber(ExampleSeparatorPerturber):
    def get_example_separator(self) -> str:
        return "\n"

class ExampleSeparatorPeriodSpacePerturber(ExampleSeparatorPerturber):
    def get_example_separator(self) -> str:
        return ". "

class ExampleSeparatorSemicolonSpacePerturber(ExampleSeparatorPerturber):
    def get_example_separator(self) -> str:
        return "; "

class ExampleSeparatorSpacePerturber(ExampleSeparatorPerturber):
    def get_example_separator(self) -> str:
        return " "

class ExampleSeparatorSpacePipeSpacePerturber(ExampleSeparatorPerturber):
    def get_example_separator(self) -> str:
        return " | "

class ExampleSeparatorTabPerturber(ExampleSeparatorPerturber):
    def get_example_separator(self) -> str:
        return "\t"

class PromptSuffixPerturber(PromptPerturber):
    pass

class PromptDropEndingPunctuationPerturber(PromptSuffixPerturber):
    def perturb(self, text: str) -> str:
        if not text:
            return text
        
        new_text = text
        if new_text[-1] in string.punctuation:
            new_text = new_text[:-1]
        
        return new_text

class PromptAddOrChangeToColonPerturber(PromptSuffixPerturber):
    def __init__(self, seed: int, perturb_prob: float) -> None:
        super().__init__(seed, perturb_prob)
        self.drop_punct = PromptDropEndingPunctuationPerturber(seed, perturb_prob)

    def perturb(self, text: str) -> str:
        return self.drop_punct.perturb(text) + ":"

class PromptAddOrChangeToHyphenPerturber(PromptSuffixPerturber):
    def __init__(self, seed: int, perturb_prob: float) -> None:
        super().__init__(seed, perturb_prob)
        self.drop_punct = PromptDropEndingPunctuationPerturber(seed, perturb_prob)

    def perturb(self, text: str) -> str:
        return self.drop_punct.perturb(text) + " -"

class TextTargetSeparatorPerturber(PromptPerturber):
    def perturb(self, text: str) -> str:
        return text

    @abstractmethod
    def get_text_target_separator(self) -> str:
        pass

class TextTargetSeparatorNewlinePerturber(TextTargetSeparatorPerturber):
    def get_text_target_separator(self) -> str:
        return "\n"

class TextTargetSeparatorTabPerturber(TextTargetSeparatorPerturber):
    def get_text_target_separator(self) -> str:
        return "\t"

class TextTargetSeparatorSpacePipeSpacePerturber(TextTargetSeparatorPerturber):
    def get_text_target_separator(self) -> str:
        return " | "

class TextTargetSeparatorRandomSpacesPerturber(TextTargetSeparatorPerturber):
    def __init__(self, seed: int, perturb_prob: float) -> None:
        super().__init__(seed, perturb_prob)
        random.seed(seed)
    
    def get_text_target_separator(self) -> str:
        return " " * random.randint(1, 5)
