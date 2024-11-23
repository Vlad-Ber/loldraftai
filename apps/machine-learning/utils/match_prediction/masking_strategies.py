from typing import Callable
import numpy as np


class LinearDecayDistribution:
    """
    Creates a linear decay distribution where probability of sampling 0 is decay_factor times
    higher than sampling max_value.
    """

    def __init__(self, max_value: int, decay_factor: float):
        self.max_value = max_value
        probabilities = np.linspace(decay_factor, 1, max_value + 1)
        self.probabilities = probabilities / probabilities.sum()

    def __call__(self) -> int:
        return np.random.choice(self.max_value + 1, p=self.probabilities)


# Dictionary mapping strategy names to their classes
MASKING_STRATEGIES = {
    "linear_decay": LinearDecayDistribution,
    # Add more strategies here
}
