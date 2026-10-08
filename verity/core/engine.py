"""The verification loop. It is the only place where the building blocks meet.

    claims -> knowledge graph -> probes -> [policy picks probe -> answer -> assess -> update belief]* -> verdict
"""
from .belief import default_pass_rates
from .types import REFUTED, SUPPORTED, UNCERTAIN, ClaimResult, SessionState, Turn, Verdict


class SessionError(Exception):
    """Raised for invalid use of a session, e.g. answering after it finished."""


class Session:
    """One verification conversation for one person.

    Usage::

        session = Session(pack, {"cv": "..."})
        while not session.finished:
            print(session.current.question)
            session.answer(input())
        print(session.verdict)
    """

    def __init__(self, pack, inputs=None, policy=None):
        self.pack = pack
        inputs = inputs or {}
        claims = pack.claim_extractor().extract(inputs)
        if not claims:
            raise ValueError("No claims found to verify.")
        graph = pack.knowledge_source().build(claims, inputs)
        probes = pack.probe_generator().generate(claims, graph)
        by_id = {claim.id: claim for claim in claims}
        for probe in probes:
            if not probe.pass_rates:
                probe.pass_rates = default_pass_rates(by_id[probe.claim_id], probe)
        belief = pack.belief_model()
        belief.start(claims)

        self.state = SessionState(claims=claims, probes=probes, belief=belief,
                                  max_questions=pack.max_questions, graph=graph, inputs=inputs)
        self.assessor = pack.assessor()
        self.policy = policy or pack.policy()
        self.policy.reset(self.state)

        self.current = None      # the probe waiting for an answer
        self.verdict = None
        self._next()

    @property
    def finished(self):
        return self.verdict is not None

    @property
    def history(self):
        return self.state.history

    def answer(self, answer):
        """Grade the answer to the current probe and move on. Returns the graded Observation."""
        if self.finished:
            raise SessionError("This session is already finished.")
        probe = self.current
        if not probe.is_free_text:
            valid = {choice.id for choice in probe.choices}
            if str(answer).strip().upper() not in valid:
                raise ValueError(f"Answer must be one of {sorted(valid)}.")
        elif not str(answer).strip():
            raise ValueError("Answer must not be empty.")

        observation = self.assessor.assess(probe, answer)
        self.state.belief.update(probe, observation)
        self.state.history.append(Turn(probe, observation))
        self.policy.observe(self.state, probe, observation)
        self._next()
        return observation

    def _next(self):
        probe = None
        if len(self.state.history) < self.state.max_questions:
            probe = self.policy.choose(self.state)
        if probe is None:
            self.current = None
            self.verdict = build_verdict(self.state)
        else:
            self.current = probe


def build_verdict(state):
    belief = state.belief
    results = []
    for claim in state.claims:
        turns = [t for t in state.history if t.probe.claim_id == claim.id]
        passed = sum(t.observation.score for t in turns)
        probability = belief.probability(claim.id)
        level = belief.level(claim.id)
        if turns:
            explanation = f"{passed:g} of {len(turns)} answers passed; {probability:.0%} likely true."
            if not claim.is_yes_no:
                explanation += (f" Most likely level: {claim.levels[level]}"
                                f" (claimed: {claim.levels[claim.claimed_level]}).")
        else:
            explanation = "Not tested."
        results.append(ClaimResult(claim, probability, belief.status(claim.id), len(turns), explanation, level))

    statuses = [r.status for r in results]
    if REFUTED in statuses:
        status = REFUTED
    elif all(s == SUPPORTED for s in statuses):
        status = SUPPORTED
    else:
        status = UNCERTAIN
    probability = min(r.probability for r in results)
    return Verdict(status, probability, results, list(state.notes))
