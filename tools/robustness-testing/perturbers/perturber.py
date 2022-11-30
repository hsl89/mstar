from abc import ABCMeta, abstractmethod
from warnings import warn

class Perturber(metaclass=ABCMeta):
    def __init__(self, seed: int, perturb_prob: float) -> None:
        """
        :param seed: Random seed
        :type seed: int
        :param perturb_prob: Controls perturbation probability in :func:`~Perturber.perturb`
        :type perturb_prob: float
        """
        self.seed = seed
        self.perturb_prob = perturb_prob
    
    @abstractmethod
    def perturb(self, text: str) -> str:
        """Perturbs text string

        :param text: Text to be perturbed
        :type text: str
        :return: Perturbed text
        :rtype: str
        """
        pass


class InputPerturber(Perturber):
    pass


class PromptPerturber(Perturber):
    def __init__(self, seed: int, perturb_prob: float) -> None:
        self.seed = seed
        self.perturb_prob = 1
        warn('PromptPerturbers always have probability of 1.')

class IdentityPerturber(Perturber):
    def __init__(self, seed: int, perturb_prob: float) -> None:
        self.seed = seed
        self.perturb_prob = 0
        warn('IdentityPerturber has perturb_prob of 0 by definition.')

    def perturb(self, text: str) -> str:
        return text
