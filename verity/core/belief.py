"""Bayesian belief about each claim.

For every claim we keep a probability for each level of its scale (two levels for a
yes/no claim). A probe says how well someone at each level does on it
(``probe.pass_rates``: the chance of passing, which is also the expected score); after an
answer, Bayes' rule re-weighs the levels.

How an answer counts (see docs/decisions/0006-score-likelihood.md):

- Multiple choice: pass or fail. Likelihood at each level is ``rate`` for a pass and
  ``1 - rate`` for a fail.
- Free text: the grader's score is "the expected score at the person's level, plus noise".
  Likelihood is a bell curve around the expected score with width ``score_noise``. So a 0.3
  on a hard question, typical for a mid-level person, supports "mid" rather than counting as
  a fail.

The claim's probability of being true is the total weight on levels at or above the
claimed level.
"""
import math

import numpy as np

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


def irt_rate(position, difficulty, discrimination=1.7, guess=0.05, slip=0.05):
    """Pass chance at a (possibly fractional) position on the level scale (see irt_pass_rates)."""
    return guess + (1 - guess - slip) / (1 + math.exp(-discrimination * (position - difficulty)))


def irt_pass_rates(n_levels, difficulty, discrimination=1.7, guess=0.05, slip=0.05):
    """Pass chance per level from item response theory (a "3-parameter logistic" curve).

    ``difficulty`` is the level (0 .. n_levels-1, may be fractional) at which a person has
    a 50/50 chance; ``discrimination`` is how sharply the chance rises around it;
    ``guess`` is the chance of passing by luck; ``slip`` the chance of failing anyway.
    """
    return [irt_rate(level, difficulty, discrimination, guess, slip) for level in range(n_levels)]


def default_pass_rates(claim, probe):
    """Pass rates for a probe that only sets p_true / p_false: levels below the claimed
    level pass with ``p_false``, the others with ``p_true``."""
    return [probe.p_true if claim.holds_at(level) else probe.p_false for level in range(len(claim.levels))]


SCORE_GRID = [i / 10 for i in range(11)]    # free-text scores considered when looking ahead


def rate_at(rates, position):
    """Pass rate at a (possibly fractional) level, by straight-line interpolation."""
    position = min(max(position, 0.0), len(rates) - 1)
    low = int(position)
    high = min(low + 1, len(rates) - 1)
    return rates[low] + (rates[high] - rates[low]) * (position - low)


def person_grid(spread, points=7):
    """Possible person offsets (in levels) and their prior weights: a normal distribution
    with standard deviation ``spread``, cut at two standard deviations. One point at 0
    when ``spread`` is 0 (no person factor)."""
    if spread <= 0:
        return [0.0], [1.0]
    offsets = [spread * (-2 + 4 * i / (points - 1)) for i in range(points)]
    weights = [math.exp(-0.5 * (o / spread) ** 2) for o in offsets]
    total = sum(weights)
    return offsets, [w / total for w in weights]


class BeliefModel:
    """Probability per level, per claim, plus an optional person factor.

    The person factor is a hidden offset (in levels) shared by all claims of one session:
    someone sharper or more tired than their level performs as if they were a bit higher or
    lower on every question. Keeping a small grid of possible offsets lets answers on one
    claim inform the others, instead of treating every answer as fully independent.
    With ``person_spread = 0`` the model is a plain per-claim model.

    ``fatigue`` is how many levels a person drops per question already answered in the
    session (0 = no drift).

    ``gap_prior`` (optional) is the base rate of "real level minus claimed level", e.g.
    ``{0: 0.6, -1: 0.2, -2: 0.15, -3: 0.05}``: how far people's real level usually is from what
    they claim. When given, it replaces ``prior`` as the starting belief (see prior_for).

    Internally, each claim's weights are a numpy array [person offset, level]; the maths is the
    same as looping over offsets and levels, just faster.
    """

    def __init__(self, prior=0.5, accept=0.9, reject=0.1, score_noise=0.2, person_spread=0.0, fatigue=0.0,
                 gap_prior=None):
        self.prior = prior       # starting chance that a claim is true
        self.accept = accept     # probability at which a claim counts as supported
        self.reject = reject     # probability at which a claim counts as refuted
        self.score_noise = score_noise       # spread of free-text scores around their expected value
        self.person_spread = person_spread   # spread of the person factor, in levels
        self.fatigue = fatigue               # levels lost per question already answered
        self.gap_prior = gap_prior           # base rate of (real level - claimed level), or None
        self.answered = 0
        self.claims = {}
        offsets, weights = person_grid(person_spread)
        self.offsets = np.array(offsets)
        self.offset_prior = np.array(weights)
        self.offset_weights = self.offset_prior.copy()
        self.weights = {}        # claim id -> array [offset, level] of level probabilities per offset

    def start(self, claims):
        self.claims = {claim.id: claim for claim in claims}
        self.answered = 0
        self.offset_weights = self.offset_prior.copy()
        self.weights = {claim.id: np.tile(self.prior_for(claim), (len(self.offsets), 1)) for claim in claims}

    def prior_for(self, claim):
        """Starting probability of each level.

        With ``gap_prior``: each level gets the base rate of its gap to the claimed level
        (a small floor for gaps never seen), renormalised.
        Otherwise: ``prior`` spread evenly over the levels where the claim holds, the rest over
        the levels below; for a yes/no claim that is simply [1 - prior, prior].
        """
        n, claimed = len(claim.levels), claim.claimed_level
        if self.gap_prior:
            weights = [self.gap_prior.get(level - claimed, 0.0) + 0.01 for level in range(n)]
            total = sum(weights)
            return [w / total for w in weights]
        if claimed == 0:
            return [1 / n] * n
        above, below = n - claimed, claimed
        return [self.prior / above if claim.holds_at(level) else (1 - self.prior) / below for level in range(n)]

    def likelihood(self, rate, score, free_text):
        """How likely ``score`` is from someone whose expected score / pass chance is ``rate``.
        Works on single numbers and on numpy arrays alike."""
        rate = np.clip(rate, MIN_RATE, 1 - MIN_RATE)
        if free_text:
            return np.exp(-((score - rate) ** 2) / (2 * self.score_noise ** 2))
        return rate ** score * (1 - rate) ** (1 - score)

    def likelihood_table(self, rates, scores, free_text):
        """Likelihood of each score, for each person offset and level, at this point of the
        session (after ``answered`` questions): an array [score, offset, level]."""
        n = len(rates)
        shift = self.offsets - self.fatigue * self.answered
        positions = np.arange(n)[None, :] + shift[:, None]                   # [offset, level]
        expected = np.interp(positions, np.arange(n), rates)                 # straight-line, clamped at the ends
        return self.likelihood(expected[None, :, :], np.array(scores, dtype=float)[:, None, None], free_text)

    def rates(self, probe):
        return probe.pass_rates or default_pass_rates(self.claims[probe.claim_id], probe)

    def update(self, probe, observation):
        score = min(max(observation.score, 0.0), 1.0)
        likes = self.likelihood_table(self.rates(probe), [score], probe.is_free_text)[0]   # [offset, level]
        new = self.weights[probe.claim_id] * likes
        evidence = new.sum(axis=1)                       # how well each offset explains the answer
        new = new / evidence[:, None] + FLOOR
        self.weights[probe.claim_id] = new / new.sum(axis=1, keepdims=True)
        self.offset_weights = self.offset_weights * evidence
        self.offset_weights = self.offset_weights / self.offset_weights.sum()
        self.answered += 1

    def level_weights(self, claim_id):
        """Probability of each level, averaged over the person offsets."""
        return list(self.offset_weights @ self.weights[claim_id])

    def holds_mask(self, claim_id):
        claim = self.claims[claim_id]
        return np.array([claim.holds_at(level) for level in range(len(claim.levels))])

    def truth_probability(self, claim_id, weights):
        return float(np.sum(np.asarray(weights)[self.holds_mask(claim_id)]))

    def probability(self, claim_id):
        """Belief that the claim is true."""
        return self.truth_probability(claim_id, self.level_weights(claim_id))

    def level(self, claim_id):
        """The most likely real level (index into the claim's levels)."""
        return int(np.argmax(self.level_weights(claim_id)))

    def person_offset(self):
        """Expected person offset in levels (0 without a person factor)."""
        return float(self.offsets @ self.offset_weights)

    def status(self, claim_id):
        p = self.probability(claim_id)
        if p >= self.accept:
            return SUPPORTED
        if p <= self.reject:
            return REFUTED
        return UNCERTAIN

    def expected_gain(self, probe):
        """How much asking ``probe`` is expected to reduce uncertainty about the claim (bits):
        current uncertainty minus the average uncertainty over the possible answers."""
        claim_id = probe.claim_id
        outcomes = SCORE_GRID if probe.is_free_text else [0.0, 1.0]
        likes = self.likelihood_table(self.rates(probe), outcomes, probe.is_free_text)   # [score, offset, level]
        mass = likes * (self.offset_weights[:, None] * self.weights[claim_id])[None, :, :]
        chance = mass.sum(axis=(1, 2))                                                 # per possible score
        holds = mass[:, :, self.holds_mask(claim_id)].sum(axis=(1, 2))
        after = sum(c / chance.sum() * entropy(h / c) for c, h in zip(chance, holds) if c > 0)
        return entropy(self.probability(claim_id)) - after
