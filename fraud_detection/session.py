"""One verification dialogue: asks the applicant questions until the policy reaches a decision."""
import random
from dataclasses import dataclass, field
from typing import List, Optional

from . import config
from .dialogue import DialogueTracker, Outcome, SystemAction
from .graph import ProcessedGraph, load_graphs, preprocess_graph
from .language import LanguageGenerator
from .policy import Policy

# Upper bound on policy steps per question, so a broken policy can never loop forever.
_MAX_STEPS_WITHOUT_QUESTION = config.MAX_EXPLORING_TIME_STEP + 1


@dataclass
class Turn:
    """What the client should show next: a question, or the final verdict."""

    status: str                                  # "question" | "fraud" | "non_fraud"
    question: Optional[str] = None
    choices: List[dict] = field(default_factory=list)   # [{"id": "A", "text": "..."}]
    answered: int = 0                            # questions answered so far
    matched: int = 0                             # of which matched the knowledge graph

    @property
    def finished(self):
        return self.status != "question"

    def to_dict(self):
        return {
            "status": self.status,
            "question": self.question,
            "choices": self.choices,
            "answered": self.answered,
            "matched": self.matched,
            "max_questions": config.MAX_EXPLORING_TIME_STEP,
        }


@dataclass
class _PendingQuestion:
    action: SystemAction
    question: str
    choices: list            # [(symbol, text)]
    q_node: int
    a_node: int


class DialogueSession:
    def __init__(self, policy: Policy, graph: ProcessedGraph, rng=random):
        self.policy = policy
        self.graph = graph
        self.tracker = DialogueTracker(graph)
        self.language = LanguageGenerator(graph, rng)
        self._node_by_name = {}
        for idx, name in graph.idx2node.items():
            self._node_by_name.setdefault(name, int(idx))

        self.history = []                        # answered questions, for replaying the chat
        self._pending: Optional[_PendingQuestion] = None
        self.outcome = Outcome.CONTINUE
        self.current = self._advance()

    @classmethod
    def from_raw_graph(cls, policy, raw_graph, rng=random):
        return cls(policy, preprocess_graph(raw_graph), rng)

    @property
    def finished(self):
        return self.outcome is not Outcome.CONTINUE

    def answer(self, choice_id: str) -> Turn:
        """Submit the applicant's answer to the pending question and move to the next turn."""
        if self.finished or self._pending is None:
            raise RuntimeError("The dialogue is already finished")
        pending = self._pending
        choice_id = str(choice_id).strip().upper()
        texts = dict(pending.choices)
        if choice_id not in texts:
            raise ValueError(f"Unknown choice {choice_id!r}; expected one of {sorted(texts)}")

        self.history.append({
            "question": pending.question,
            "choices": [{"id": symbol, "text": text} for symbol, text in pending.choices],
            "selected": choice_id,
        })
        # "Not Sure" (or anything that is not a graph node) can never match the correct node.
        answered_node = self._node_by_name.get(texts[choice_id], -1)
        self._pending = None
        self.outcome = self.tracker.finish_step(
            pending.action, answered_node, (pending.q_node, pending.a_node))
        self.current = self._advance()
        return self.current

    def _advance(self) -> Turn:
        """Run the policy until it either asks a question or reaches a final decision."""
        if self.finished:
            return self._turn(self.outcome.value)

        for _ in range(_MAX_STEPS_WITHOUT_QUESTION):
            action = self.policy.decide(self.graph, self.tracker.state)
            self.tracker.begin_step(action)

            if action.role == "worker" and not action.is_terminal:
                q_node = self.graph.personal_nodes[self.tracker.state.active_worker]
                question, choices = self.language.generate(q_node, action.content)
                self._pending = _PendingQuestion(action, question, choices, q_node, action.content)
                return self._turn("question", question, choices)

            # The manager moved on to another topic, or somebody made a decision: nothing to ask.
            self.outcome = self.tracker.finish_step(action, None, (None, None))
            if self.finished:
                return self._turn(self.outcome.value)
        raise RuntimeError("The policy did not produce a question or a decision")

    def _turn(self, status, question=None, choices=()):
        state = self.tracker.state
        return Turn(
            status=status,
            question=question,
            choices=[{"id": symbol, "text": text} for symbol, text in choices],
            answered=state.total_qa_turn,
            matched=len(state.known_nodes),
        )


class Engine:
    """Loads the trained models and graphs once; creates independent dialogue sessions."""

    def __init__(self, policy: Optional[Policy] = None, graphs: Optional[list] = None, rng=random):
        self.policy = policy or Policy()
        self.graphs = graphs if graphs is not None else load_graphs()
        self.rng = rng

    def new_session(self, graph_index: Optional[int] = None) -> DialogueSession:
        raw = self.rng.choice(self.graphs) if graph_index is None else self.graphs[graph_index]
        return DialogueSession.from_raw_graph(self.policy, raw, self.rng)
