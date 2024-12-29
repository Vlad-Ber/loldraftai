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


class StrategicMaskingDistribution:
    """
    Probablilities are determined by my instinct, but seem abount right.
    Strategic masking with specific scenarios:
    - 10% chance to mask one full team (5 champions), to learn champion synergies
    - 0.5% chance to mask everything (10 champions), to have a baseline model output
    - 20% chance to mask nothing (0 champions), to learn complex champion interactions, with full visibility
    - 69.5% chance to mask 1-9 champions with linear decay, to learn partial drafts
    """

    def __init__(self, decay_factor: float = 2.0):
        self.max_value = 10  # Total number of champions that can be masked

        # Initialize probabilities array
        self.probabilities = np.zeros(self.max_value + 1)

        # Set fixed scenario probabilities
        self.probabilities[0] = 0.20  # No masking
        self.probabilities[5] = 0.10  # One full team
        self.probabilities[10] = 0.005  # Everything masked

        # Distribute remaining probability (0.695) across 1-4, 6-9 using linear decay
        remaining_indices = [i for i in range(1, 10) if i != 5]
        linear_probs = np.linspace(decay_factor, 1, len(remaining_indices))
        linear_probs = linear_probs / linear_probs.sum() * 0.695

        # Assign the linearly decaying probabilities
        for idx, prob in zip(remaining_indices, linear_probs):
            self.probabilities[idx] = prob

        # Normalize to ensure sum is exactly 1
        self.probabilities = self.probabilities / self.probabilities.sum()

    def __call__(self) -> int:
        return np.random.choice(self.max_value + 1, p=self.probabilities)

    def get_distribution(self) -> np.ndarray:
        """Returns the probability distribution for debugging/visualization"""
        return self.probabilities


# Dictionary mapping strategy names to their classes
MASKING_STRATEGIES = {
    "linear_decay": LinearDecayDistribution,
    "strategic": StrategicMaskingDistribution,
}
