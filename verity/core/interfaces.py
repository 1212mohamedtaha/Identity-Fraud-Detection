"""Base classes for the building blocks. A domain pack subclasses the ones it needs.

Each method documents its contract; see docs/specs/core.md for the full picture.
"""
from .types import Observation


class ClaimExtractor:
    """Turns raw user input (a form, a CV, ...) into a list of Claim objects."""

    def extract(self, inputs):
        raise NotImplementedError


class KnowledgeSource:
    """Builds the KnowledgeGraph of facts that can back or refute the claims."""

    def build(self, claims, inputs):
        raise NotImplementedError


class ProbeGenerator:
    """Creates the candidate probes (questions) for the claims."""

    def generate(self, claims, graph):
        raise NotImplementedError


class Assessor:
    """Grades one answer to one probe into an Observation."""

    def assess(self, probe, answer):
        raise NotImplementedError


class ChoiceAssessor(Assessor):
    """Default assessor for multiple-choice probes: 1.0 if the right option was picked."""

    def assess(self, probe, answer):
        picked = str(answer).strip().upper()
        score = 1.0 if picked == probe.answer.upper() else 0.0
        return Observation(probe.id, probe.claim_id, picked, score)


class Policy:
    """Chooses the next probe to ask, or None to stop.

    ``reset`` is called once when a session starts and ``observe`` after every answer,
    so stateful policies can keep their own bookkeeping.
    """

    name = "policy"

    def reset(self, state):
        pass

    def choose(self, state):
        raise NotImplementedError

    def observe(self, state, probe, observation):
        pass


class Respondent:
    """A simulated person answering probes; used for training and evaluation only."""

    def answer(self, probe):
        raise NotImplementedError
