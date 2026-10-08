"""Plain data objects passed between the building blocks. See docs/specs/core.md."""
from dataclasses import dataclass, field
from typing import Optional

SUPPORTED = "supported"
REFUTED = "refuted"
UNCERTAIN = "uncertain"

YES_NO = ("false", "true")


@dataclass
class Claim:
    """Something a person asserts, e.g. "Works at ITI Mansoura" or "Knows SQL at senior level".

    Every claim lives on an ordered scale of ``levels``. A yes/no claim is the two-level
    scale ("false", "true"); a skill can use ("none", "beginner", ..., "senior").
    The claim holds when the person's real level is at or above ``claimed_level``
    (an index into ``levels``; defaults to the top level).
    """

    id: str
    text: str
    kind: str = ""                          # free label, e.g. "company" or "skill"
    data: dict = field(default_factory=dict)  # anything the pack needs later
    levels: tuple = YES_NO
    claimed_level: Optional[int] = None

    def __post_init__(self):
        if self.claimed_level is None:
            self.claimed_level = len(self.levels) - 1

    @property
    def is_yes_no(self):
        return tuple(self.levels) == YES_NO

    def holds_at(self, level):
        """Is the claim true for someone whose real level is ``level``?"""
        return level >= self.claimed_level


@dataclass
class Choice:
    id: str      # "A", "B", ...
    text: str


@dataclass
class Probe:
    """One question (or task) that tests one claim.

    ``pass_rates[level]`` is the chance that someone at that level of the claim's scale
    passes this probe; the belief model uses it to weigh each answer. For a yes/no claim
    it is simply ``[p_false, p_true]``, so packs may set just ``p_true`` / ``p_false``
    and the engine fills ``pass_rates`` in (see belief.default_pass_rates).
    """

    id: str
    claim_id: str
    question: str
    choices: list = field(default_factory=list)   # list[Choice]; empty means a free-text answer
    answer: str = ""          # correct choice id, or a reference answer for free text
    rubric: str = ""          # grading guide for free-text answers
    difficulty: str = "medium"
    p_true: float = 0.8
    p_false: float = 0.25
    pass_rates: list = field(default_factory=list)
    data: dict = field(default_factory=dict)

    @property
    def is_free_text(self):
        return not self.choices


@dataclass
class Observation:
    """The graded result of answering one probe."""

    probe_id: str
    claim_id: str
    answer: str
    score: float              # 0.0 (failed) .. 1.0 (passed)
    feedback: str = ""


@dataclass
class ClaimResult:
    claim: Claim
    probability: float       # belief that the claim is true
    status: str              # SUPPORTED / REFUTED / UNCERTAIN
    questions: int
    explanation: str
    level: Optional[int] = None   # most likely real level (index into claim.levels)


@dataclass
class Verdict:
    status: str
    probability: float       # belief that *every* claim is true (the weakest claim)
    claims: list             # list[ClaimResult]
    notes: list = field(default_factory=list)   # extra remarks, e.g. what a policy concluded


@dataclass
class Turn:
    """One answered question, kept for the transcript and the report."""

    probe: Probe
    observation: Observation


@dataclass
class SessionState:
    """Everything a policy may look at when choosing the next probe."""

    claims: list                  # list[Claim]
    probes: list                  # list[Probe], all candidate probes
    belief: object                # BeliefModel
    max_questions: int
    graph: object = None          # KnowledgeGraph
    inputs: dict = field(default_factory=dict)
    history: list = field(default_factory=list)   # list[Turn]
    notes: list = field(default_factory=list)
    data: dict = field(default_factory=dict)      # free space for packs and policies

    def asked_ids(self):
        return {turn.probe.id for turn in self.history}

    def unasked(self):
        asked = self.asked_ids()
        return [probe for probe in self.probes if probe.id not in asked]

    def questions_for(self, claim_id):
        return sum(1 for turn in self.history if turn.probe.claim_id == claim_id)

    def probe(self, probe_id) -> Optional[Probe]:
        return next((p for p in self.probes if p.id == probe_id), None)
