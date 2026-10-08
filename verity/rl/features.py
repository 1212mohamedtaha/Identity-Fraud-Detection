"""Numbers describing the session, fed to the policy network.

Features are domain-agnostic (they use beliefs, levels and probe pass rates only), so one
network design works for every pack.
"""
from ..core.belief import rate_at
from ..core.types import UNCERTAIN

PROBE_FEATURES = 12
STATE_FEATURES = 7


def mean_score(state, claim_id):
    scores = [t.observation.score for t in state.history if t.probe.claim_id == claim_id]
    return sum(scores) / len(scores) if scores else 0.5


def probe_features(state, probe, gain=None):
    """``gain``: the probe's expected gain, if already computed."""
    belief = state.belief
    claim = belief.claims[probe.claim_id]
    p = belief.probability(probe.claim_id)
    rates = belief.rates(probe)
    claimed = claim.claimed_level
    top = max(len(claim.levels) - 1, 1)
    # How sharply the probe separates "holds" (claimed level) from "just below it".
    separation = rate_at(rates, claimed) - rate_at(rates, max(claimed - 1, 0))
    return [
        p,                                                     # belief the claim is true
        1 - abs(p - 0.5) * 2,                                  # how undecided the claim is (1 = 50/50)
        belief.expected_gain(probe) if gain is None else gain,   # information this probe should give
        separation,
        rate_at(rates, claimed),                               # expected score at the claimed level
        claimed / top,
        state.questions_for(probe.claim_id) / state.max_questions,
        mean_score(state, probe.claim_id),
        len(state.history) / state.max_questions,
        1.0 if belief.status(probe.claim_id) == UNCERTAIN else 0.0,
        1.0 if probe.is_free_text else 0.0,
        belief.person_offset(),
    ]


def state_features(state, gains=None):
    """Summary of the whole session: used to score "stop" and to predict the outcome.
    ``gains``: expected gains of the open probes, if already computed."""
    belief = state.belief
    claims = state.claims
    probabilities = [belief.probability(c.id) for c in claims]
    decided = sum(1 for c in claims if belief.status(c.id) != UNCERTAIN) / len(claims)
    if gains is None:
        gains = [belief.expected_gain(p) for p in state.unasked() if belief.status(p.claim_id) == UNCERTAIN]
    return [
        decided,
        len(state.history) / state.max_questions,
        min(probabilities),
        sum(probabilities) / len(probabilities),
        sum(1 - abs(p - 0.5) * 2 for p in probabilities) / len(probabilities),
        max(gains, default=0.0),
        belief.person_offset(),
    ]
