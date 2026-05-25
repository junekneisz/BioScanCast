from __future__ import annotations

from typing import Sequence

import numpy as np

EPSILON = 1e-15


def _to_probability_vector(probabilities: Sequence[float]) -> np.ndarray:
    """Convert a sequence of raw values into a valid probability vector."""
    probs = np.asarray(probabilities, dtype=float)

    if probs.ndim != 1:
        raise ValueError("Probabilities must be a one-dimensional sequence.")
    if len(probs) == 0:
        raise ValueError("Probabilities cannot be empty.")
    if np.any(probs < 0):
        raise ValueError("Probabilities cannot contain negative values.")

    total = probs.sum()
    if total <= 0:
        raise ValueError("Probabilities must sum to a positive value.")

    return probs / total


def multiclass_brier_score(probabilities: Sequence[float], true_index: int) -> float:
    """Compute the multiclass Brier score. Lower is better."""
    probs = _to_probability_vector(probabilities)
    if true_index < 0 or true_index >= len(probs):
        raise IndexError("true_index is out of bounds for the probability vector.")

    outcome = np.zeros(len(probs), dtype=float)
    outcome[true_index] = 1.0
    return float(np.sum((probs - outcome) ** 2))


def binary_brier_score(probability_yes: float, outcome_yes: int) -> float:
    """Compute the Brier score for a binary YES/NO forecast."""
    p_yes = float(probability_yes)
    if p_yes < 0:
        raise ValueError("Probability cannot be negative.")
    if p_yes > 1:
        p_yes = p_yes / 100.0
    p_yes = min(max(p_yes, 0.0), 1.0)
    if outcome_yes not in (0, 1):
        raise ValueError("outcome_yes must be either 0 or 1.")
    return float((p_yes - outcome_yes) ** 2)


def log_score(probabilities: Sequence[float], true_index: int) -> float:
    """Compute the logarithmic score for a multiclass forecast. Lower is better."""
    probs = _to_probability_vector(probabilities)
    if true_index < 0 or true_index >= len(probs):
        raise IndexError("true_index is out of bounds for the probability vector.")

    p_true = float(probs[true_index])
    p_true = np.clip(p_true, EPSILON, 1.0 - EPSILON)
    return float(-np.log(p_true))


def binary_log_score(probability_yes: float, outcome_yes: int) -> float:
    """Compute the log score for a binary YES/NO forecast."""
    p_yes = float(probability_yes)
    if p_yes < 0:
        raise ValueError("Probability cannot be negative.")
    if p_yes > 1:
        p_yes = p_yes / 100.0
    p_yes = min(max(p_yes, 0.0), 1.0)
    p_no = 1.0 - p_yes
    if outcome_yes not in (0, 1):
        raise ValueError("outcome_yes must be either 0 or 1.")

    p_true = p_yes if outcome_yes == 1 else p_no
    p_true = np.clip(p_true, EPSILON, 1.0 - EPSILON)
    return float(-np.log(p_true))


def accuracy(probabilities: Sequence[float], true_index: int) -> int:
    """Return 1 if the most likely bucket matches the resolved bucket."""
    probs = _to_probability_vector(probabilities)
    if true_index < 0 or true_index >= len(probs):
        raise IndexError("true_index is out of bounds for the probability vector.")
    return int(int(np.argmax(probs)) == true_index)


def ranked_probability_score(probabilities: Sequence[float], true_index: int) -> float:
    """Ranked Probability Score for ordered buckets. Lower is better."""
    probs = _to_probability_vector(probabilities)
    if true_index < 0 or true_index >= len(probs):
        raise IndexError("true_index is out of bounds for the probability vector.")

    outcome = np.zeros(len(probs), dtype=float)
    outcome[true_index] = 1.0
    cum_probs = np.cumsum(probs)
    cum_outcome = np.cumsum(outcome)
    return float(np.sum((cum_probs[:-1] - cum_outcome[:-1]) ** 2) / (len(probs) - 1))


def top_probability(probabilities: Sequence[float]) -> float:
    """Return the largest assigned probability (forecast sharpness)."""
    probs = _to_probability_vector(probabilities)
    return float(np.max(probs))


def true_probability(probabilities: Sequence[float], true_index: int) -> float:
    """Return the probability assigned to the true bucket."""
    probs = _to_probability_vector(probabilities)
    if true_index < 0 or true_index >= len(probs):
        raise IndexError("true_index is out of bounds for the probability vector.")
    return float(probs[true_index])


def entropy(probabilities: Sequence[float]) -> float:
    """Shannon entropy in nats."""
    probs = _to_probability_vector(probabilities)
    probs = np.clip(probs, EPSILON, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def normalized_entropy(probabilities: Sequence[float]) -> float:
    """Entropy scaled to [0, 1] for easier comparison across questions."""
    probs = _to_probability_vector(probabilities)
    if len(probs) <= 1:
        return 0.0
    return float(entropy(probs) / np.log(len(probs)))
