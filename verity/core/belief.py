"""Bayesian belief about each claim being true.

For every claim we keep log-odds. A probe with pass rates ``p_true`` (claim true) and
``p_false`` (claim false) moves the log-odds by the log likelihood ratio of the answer.
A score is read as "the chance the answer passed": 1.0 is a pass, 0.0 a fail, 0.5 is
neutral, and a decent free-text answer (say 0.7) counts as mild evidence for the claim.
"""
import math

from .types import REFUTED, SUPPORTED, UNCERTAIN

MIN_P = 0.01   # keep probabilities away from 0 and 1 so one answer never decides everything
MAX_P = 0.99


def clamp(p):
    return min(max(p, MIN_P), MAX_P)


def logit(p):
    p = clamp(p)
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def entropy(p):
    """Uncertainty of a yes/no belief, in bits (1.0 = no idea, 0.0 = certain)."""
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


class BeliefModel:
    """Belief about each claim. Subclass to use a different statistical model."""

    def __init__(self, prior=0.5, accept=0.9, reject=0.1):
        self.prior = prior
        self.accept = accept     # probability at which a claim counts as supported
        self.reject = reject     # probability at which a claim counts as refuted
        self.log_odds = {}

    def start(self, claims):
        self.log_odds = {claim.id: logit(self.prior) for claim in claims}

    def update(self, probe, observation):
        p_true, p_false = clamp(probe.p_true), clamp(probe.p_false)
        s = min(max(observation.score, 0.0), 1.0)
        if_true = s * p_true + (1 - s) * (1 - p_true)      # chance of this result if the claim is true
        if_false = s * p_false + (1 - s) * (1 - p_false)   # ... and if it is false
        change = math.log(if_true / if_false)
        self.log_odds[probe.claim_id] = logit(sigmoid(self.log_odds[probe.claim_id] + change))

    def probability(self, claim_id):
        return sigmoid(self.log_odds[claim_id])

    def status(self, claim_id):
        p = self.probability(claim_id)
        if p >= self.accept:
            return SUPPORTED
        if p <= self.reject:
            return REFUTED
        return UNCERTAIN

    def expected_gain(self, probe):
        """How much asking ``probe`` is expected to reduce uncertainty (in bits)."""
        p = self.probability(probe.claim_id)
        p_true, p_false = clamp(probe.p_true), clamp(probe.p_false)
        p_pass = p * p_true + (1 - p) * p_false
        p_if_pass = p * p_true / p_pass
        p_if_fail = p * (1 - p_true) / (1 - p_pass)
        return entropy(p) - (p_pass * entropy(p_if_pass) + (1 - p_pass) * entropy(p_if_fail))
