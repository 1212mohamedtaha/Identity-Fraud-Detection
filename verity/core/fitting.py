"""Fit question parameters from data, so pass rates are measured instead of guessed.

Given answers to one question from people whose real level is known, find the item
response theory curve (difficulty, discrimination) that explains them best, by trying
every combination on a grid. Simple and transparent; fast enough for thousands of answers.
"""
import math

from .belief import clamp, irt_pass_rates

DISCRIMINATIONS = (0.8, 1.2, 1.7, 2.4, 3.2)


def log_likelihood(observations, rates):
    """``observations``: list of (level, score). Scores between 0 and 1 count partly as a pass."""
    total = 0.0
    for level, score in observations:
        rate = clamp(rates[level])
        total += score * math.log(rate) + (1 - score) * math.log(1 - rate)
    return total


def fit_item(observations, n_levels, step=0.1):
    """Best (difficulty, discrimination) for one question. Difficulty is searched from
    -1 to n_levels in ``step`` increments."""
    best, best_ll = None, -math.inf
    steps = int((n_levels + 1) / step) + 1
    for discrimination in DISCRIMINATIONS:
        for i in range(steps):
            difficulty = -1 + i * step
            ll = log_likelihood(observations, irt_pass_rates(n_levels, difficulty, discrimination))
            if ll > best_ll:
                best, best_ll = (round(difficulty, 2), discrimination), ll
    return best
