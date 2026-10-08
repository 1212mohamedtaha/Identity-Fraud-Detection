"""Fit question parameters from data, so pass rates are measured instead of guessed.

Given answers to one question from people whose real level is known, find the item
response theory curve (difficulty, discrimination) that explains them best, by trying
every combination on a grid. Simple and transparent; fast enough for thousands of answers.
"""
import math

from .belief import clamp, irt_rate

DISCRIMINATIONS = (0.8, 1.2, 1.7, 2.4, 3.2)


def score_log_likelihood(rate, score):
    """Log-likelihood of a score (0..1; partial scores count partly as a pass)."""
    rate = clamp(rate)
    return score * math.log(rate) + (1 - score) * math.log(1 - rate)


def fit_item(observations, n_levels, step=0.1):
    """Best (difficulty, discrimination) for one question.

    ``observations``: list of (position, score); position is the person's level, possibly
    adjusted (fractional) for their person offset and fatigue. Difficulty is searched from
    -1 to n_levels in ``step`` increments."""
    best, best_ll = None, -math.inf
    steps = int((n_levels + 1) / step) + 1
    for discrimination in DISCRIMINATIONS:
        for i in range(steps):
            difficulty = -1 + i * step
            ll = sum(score_log_likelihood(irt_rate(x, difficulty, discrimination), s) for x, s in observations)
            if ll > best_ll:
                best, best_ll = (round(difficulty, 2), discrimination), ll
    return best


OFFSETS = [round(-1.5 + 0.1 * i, 2) for i in range(31)]          # person offsets tried, in levels
FATIGUES = [round(0.01 * i, 2) for i in range(9)]                 # drift per question tried


def person_log_likelihood(answers, offset, fatigue):
    """``answers``: list of (level, position_in_session, score, item) with item = (difficulty,
    discrimination). Log-likelihood of all answers for a person with ``offset``."""
    return sum(score_log_likelihood(irt_rate(level + offset - fatigue * position, *item), score)
               for level, position, score, item in answers)


def fit_person_offset(answers, fatigue=0.0):
    """Most likely person offset (in levels) for one person's answers."""
    return max(OFFSETS, key=lambda offset: person_log_likelihood(answers, offset, fatigue))


def fit_fatigue(people):
    """Drift per question that best explains everyone's answers, each person getting their own
    best offset. ``people``: list of answer lists (see person_log_likelihood)."""
    def total(fatigue):
        return sum(max(person_log_likelihood(a, o, fatigue) for o in OFFSETS) for a in people)
    return max(FATIGUES, key=total)


def person_spread(people, fatigue=0.0):
    """Spread (standard deviation, in levels) of the true person offsets.

    Each person's offset is estimated twice, from two halves of their answers. The two
    estimates share only the true offset (their estimation errors are independent), so their
    covariance estimates the true variance without the estimation noise.
    """
    first = [fit_person_offset(a[0::2], fatigue) for a in people]
    second = [fit_person_offset(a[1::2], fatigue) for a in people]
    n = len(people)
    mean1, mean2 = sum(first) / n, sum(second) / n
    covariance = sum((x - mean1) * (y - mean2) for x, y in zip(first, second)) / n
    return math.sqrt(max(covariance, 0.0))
