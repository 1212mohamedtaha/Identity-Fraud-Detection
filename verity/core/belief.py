"""Bayesian belief about each claim.

For every claim we keep a probability for each level of its scale (two levels for a
yes/no claim). A probe says how likely someone at each level is to pass it
(``probe.pass_rates``); after an answer, Bayes' rule re-weighs the levels.

A score is read as "the chance the answer passed": 1.0 is a pass, 0.0 a fail, 0.5 is
neutral, and a decent free-text answer (say 0.7) counts as mild evidence of a higher level.

The claim's probability of being true is the total weight on levels at or above the
claimed level.
"""
import math

from .types import REFUTED, SUPPORTED, UNCERTAIN

MIN_RATE = 0.01     # no answer is ever treated as completely impossible
FLOOR = 0.001       # every level keeps a little weight, so one answer never decides everything


def clamp(p, low=MIN_RATE, high=1 - MIN_RATE):
    return min(max(p, low), high)


def entropy(p):
    """Uncertainty of a yes/no belief, in bits (1.0 = no idea, 0.0 = certain)."""
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def irt_pass_rates(n_levels, difficulty, discrimination=1.7, guess=0.05, slip=0.05):
    """Pass chance per level from item response theory (a "3-parameter logistic" curve).

    ``difficulty`` is the level (0 .. n_levels-1, may be fractional) at which a person has
    a 50/50 chance; ``discrimination`` is how sharply the chance rises around it;
    ``guess`` is the chance of passing by luck; ``slip`` the chance of failing anyway.
    """
    rates = []
    for level in range(n_levels):
        curve = 1 / (1 + math.exp(-discrimination * (level - difficulty)))
        rates.append(guess + (1 - guess - slip) * curve)
    return rates


def default_pass_rates(claim, probe):
    """Pass rates for a probe that only sets p_true / p_false: levels below the claimed
    level pass with ``p_false``, the others with ``p_true``."""
    return [probe.p_true if claim.holds_at(level) else probe.p_false for level in range(len(claim.levels))]


class BeliefModel:
    """Probability per level, per claim. Subclass to use a different statistical model."""

    def __init__(self, prior=0.5, accept=0.9, reject=0.1):
        self.prior = prior       # starting chance that a claim is true
        self.accept = accept     # probability at which a claim counts as supported
        self.reject = reject     # probability at which a claim counts as refuted
        self.claims = {}
        self.weights = {}        # claim id -> list of probabilities, one per level

    def start(self, claims):
        self.claims = {claim.id: claim for claim in claims}
        self.weights = {claim.id: self.prior_for(claim) for claim in claims}

    def prior_for(self, claim):
        """``prior`` spread evenly over the levels where the claim holds, the rest over the
        levels below. For a yes/no claim that is simply [1 - prior, prior]."""
        n, claimed = len(claim.levels), claim.claimed_level
        if claimed == 0:
            return [1 / n] * n
        above, below = n - claimed, claimed
        return [self.prior / above if claim.holds_at(level) else (1 - self.prior) / below for level in range(n)]

    def posterior(self, weights, rates, score):
        """New level weights after an answer with ``score`` to a probe with ``rates``."""
        s = min(max(score, 0.0), 1.0)
        new = []
        for weight, rate in zip(weights, rates):
            rate = clamp(rate)
            new.append(weight * (s * rate + (1 - s) * (1 - rate)))
        total = sum(new)
        new = [w / total + FLOOR for w in new]
        total = sum(new)
        return [w / total for w in new]

    def rates(self, probe):
        return probe.pass_rates or default_pass_rates(self.claims[probe.claim_id], probe)

    def update(self, probe, observation):
        self.weights[probe.claim_id] = self.posterior(
            self.weights[probe.claim_id], self.rates(probe), observation.score)

    def truth_probability(self, claim_id, weights):
        claim = self.claims[claim_id]
        return sum(w for level, w in enumerate(weights) if claim.holds_at(level))

    def probability(self, claim_id):
        """Belief that the claim is true."""
        return self.truth_probability(claim_id, self.weights[claim_id])

    def level(self, claim_id):
        """The most likely real level (index into the claim's levels)."""
        weights = self.weights[claim_id]
        return max(range(len(weights)), key=weights.__getitem__)

    def status(self, claim_id):
        p = self.probability(claim_id)
        if p >= self.accept:
            return SUPPORTED
        if p <= self.reject:
            return REFUTED
        return UNCERTAIN

    def expected_gain(self, probe):
        """How much asking ``probe`` is expected to reduce uncertainty about the claim (bits)."""
        weights, rates = self.weights[probe.claim_id], self.rates(probe)
        p_pass = sum(w * clamp(r) for w, r in zip(weights, rates))
        if_pass = self.truth_probability(probe.claim_id, self.posterior(weights, rates, 1.0))
        if_fail = self.truth_probability(probe.claim_id, self.posterior(weights, rates, 0.0))
        now = self.probability(probe.claim_id)
        return entropy(now) - (p_pass * entropy(if_pass) + (1 - p_pass) * entropy(if_fail))
