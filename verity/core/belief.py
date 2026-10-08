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
        self.offsets, self.offset_prior = person_grid(person_spread)
        self.offset_weights = list(self.offset_prior)
        self.weights = {}        # claim id -> one list of level probabilities per person offset

    def start(self, claims):
        self.claims = {claim.id: claim for claim in claims}
        self.answered = 0
        self.offset_weights = list(self.offset_prior)
        self.weights = {claim.id: [self.prior_for(claim) for _ in self.offsets] for claim in claims}

    def prior_for(self, claim):
        """Starting probability of each level.

        With ``gap_prior``: each level gets the base rate of its gap to the claimed level
        (a small floor for gaps never seen), renormalised over the claim's levels.
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
        """How likely ``score`` is from someone whose expected score / pass chance is ``rate``."""
        rate = clamp(rate)
        if free_text:
            return math.exp(-((score - rate) ** 2) / (2 * self.score_noise ** 2))
        return rate ** score * (1 - rate) ** (1 - score)

    def level_likelihoods(self, rates, offset, score, free_text):
        """Likelihood of ``score`` at every level, for a person with ``offset``, at this point
        of the session (after ``answered`` questions)."""
        shift = offset - self.fatigue * self.answered
        return [self.likelihood(rate_at(rates, level + shift), score, free_text) for level in range(len(rates))]

    def rates(self, probe):
        return probe.pass_rates or default_pass_rates(self.claims[probe.claim_id], probe)

    def update(self, probe, observation):
        rates, score = self.rates(probe), min(max(observation.score, 0.0), 1.0)
        for k, offset in enumerate(self.offsets):
            old = self.weights[probe.claim_id][k]
            likes = self.level_likelihoods(rates, offset, score, probe.is_free_text)
            new = [w * like for w, like in zip(old, likes)]
            evidence = sum(new)                       # how well this offset explains the answer
            new = [w / evidence + FLOOR for w in new]
            total = sum(new)
            self.weights[probe.claim_id][k] = [w / total for w in new]
            self.offset_weights[k] *= evidence
        total = sum(self.offset_weights)
        self.offset_weights = [w / total for w in self.offset_weights]
        self.answered += 1

    def level_weights(self, claim_id):
        """Probability of each level, averaged over the person offsets."""
        per_offset = self.weights[claim_id]
        return [sum(pk * levels[i] for pk, levels in zip(self.offset_weights, per_offset))
                for i in range(len(per_offset[0]))]

    def truth_probability(self, claim_id, weights):
        claim = self.claims[claim_id]
        return sum(w for level, w in enumerate(weights) if claim.holds_at(level))

    def probability(self, claim_id):
        """Belief that the claim is true."""
        return self.truth_probability(claim_id, self.level_weights(claim_id))

    def level(self, claim_id):
        """The most likely real level (index into the claim's levels)."""
        weights = self.level_weights(claim_id)
        return max(range(len(weights)), key=weights.__getitem__)

    def person_offset(self):
        """Expected person offset in levels (0 without a person factor)."""
        return sum(o * w for o, w in zip(self.offsets, self.offset_weights))

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
        claim_id, rates = probe.claim_id, self.rates(probe)
        outcomes = SCORE_GRID if probe.is_free_text else [0.0, 1.0]
        joint = []           # per outcome: (chance, chance the claim holds afterwards)
        for score in outcomes:
            total = holds = 0.0
            for pk, offset, levels in zip(self.offset_weights, self.offsets, self.weights[claim_id]):
                likes = self.level_likelihoods(rates, offset, score, probe.is_free_text)
                for level, (w, like) in enumerate(zip(levels, likes)):
                    mass = pk * w * like
                    total += mass
                    if self.claims[claim_id].holds_at(level):
                        holds += mass
            joint.append((total, holds / total if total else 0.0))
        norm = sum(chance for chance, _ in joint)
        after = sum(chance / norm * entropy(p) for chance, p in joint)
        return entropy(self.probability(claim_id)) - after
