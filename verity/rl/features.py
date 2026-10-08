"""Numbers describing the session, fed to the policy network.

Features are domain-agnostic (they only use beliefs and probe pass rates), so one
network design works for every pack.
"""
from ..core.types import UNCERTAIN

PROBE_FEATURES = 8
STOP_FEATURES = 4


def probe_features(state, probe):
    belief = state.belief
    p = belief.probability(probe.claim_id)
    return [
        p,                                                     # belief the claim is true
        1 - abs(p - 0.5) * 2,                                  # how undecided the claim is (1 = 50/50)
        belief.expected_gain(probe),                           # information this probe should give
        probe.p_true,
        probe.p_false,
        state.questions_for(probe.claim_id) / state.max_questions,
        len(state.history) / state.max_questions,
        1.0 if belief.status(probe.claim_id) == UNCERTAIN else 0.0,
    ]


def stop_features(state):
    belief = state.belief
    claims = state.claims
    decided = sum(1 for c in claims if belief.status(c.id) != UNCERTAIN) / len(claims)
    gains = [belief.expected_gain(p) for p in state.unasked()]
    return [
        decided,
        len(state.history) / state.max_questions,
        min(belief.probability(c.id) for c in claims),
        max(gains, default=0.0),
    ]
