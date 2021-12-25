"""
For end applications it is sometimes for useful to talk confidences rather than probabilities. Confidences directly
tell users that, if the probability of class 1 was 0.01, we can be quite confident the class is 0.

Remember that any interpretation of probabilities depends on the probabilities being properly calibrated.
"""
from typing import List


def binary_classification_confidence(probabilities: List[float]) -> List[float]:
    """
    Computes confidences (c) from predicted probabilities (p), following the formula:
    if p < 0.5 -> c = 2p - 1
    else -> c = 1 - 2p

    This effectively makes that values closer to absolute 0 or 1 have higher confidences, and values closer to 0.5
    have smaller confidences.

    Note that the prediction threshold is assumed fixed at 0.5.
    """
    return list(map(lambda p: 2 * p - 1 if p >= 0.5 else 1 - 2 * p, probabilities))
