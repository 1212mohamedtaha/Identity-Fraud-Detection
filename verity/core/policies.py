"""Built-in policies. A policy picks the next probe or returns None to stop."""
import random

from .interfaces import Policy
from .types import REFUTED, UNCERTAIN


def open_candidates(state):
    """Unasked probes whose claim is still undecided."""
    return [p for p in state.unasked() if state.belief.status(p.claim_id) == UNCERTAIN]


def verdict_settled(state):
    """True once more questions cannot change the overall verdict: one refuted claim
    refutes the whole set, so there is nothing left to learn."""
    return any(state.belief.status(c.id) == REFUTED for c in state.claims)


class GreedyPolicy(Policy):
    """Focus on the weakest undecided claim and ask its most informative probe.

    One false claim is enough to refute the whole set, so digging into the most
    doubtful claim first reaches a verdict sooner than spreading questions evenly.
    Stops once the verdict is settled: every claim decided, or any claim refuted.
    """

    name = "greedy"

    def choose(self, state):
        if verdict_settled(state):
            return None
        candidates = open_candidates(state)
        if not candidates:
            return None
        belief = state.belief
        weakest = min(belief.probability(p.claim_id) for p in candidates)
        focus = [p for p in candidates if belief.probability(p.claim_id) == weakest]
        return max(focus, key=belief.expected_gain)


class RandomPolicy(Policy):
    """Ask a random open probe. Useful only as a baseline in evaluations."""

    name = "random"

    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def choose(self, state):
        candidates = open_candidates(state)
        return self.rng.choice(candidates) if candidates else None
