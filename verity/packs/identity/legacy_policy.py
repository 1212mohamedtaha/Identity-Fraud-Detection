"""Runs the original pretrained hierarchical RL model as a Verity policy."""
from functools import lru_cache

from ...core.interfaces import Policy
from .legacy import config
from .legacy.dialogue import DialogueTracker, Outcome
from .legacy.policy import Policy as LegacyModel


@lru_cache(maxsize=None)
def legacy_model():
    return LegacyModel()      # loads the checkpoint once per process


class LegacyRLPolicy(Policy):
    """The manager picks a personal attribute, its worker picks a place to ask about.

    The model keeps its own dialogue state; when it reaches its own fraud / not-fraud
    decision the session stops and the decision is added to the verdict notes.
    """

    name = "legacy"

    def reset(self, state):
        self.graph = state.graph.data["processed"]
        self.tracker = DialogueTracker(self.graph)
        self.pending = None

    def choose(self, state):
        for _ in range(config.MAX_EXPLORING_TIME_STEP + 1):
            action = legacy_model().decide(self.graph, self.tracker.state)
            self.tracker.begin_step(action)

            if action.role == "worker" and not action.is_terminal:
                q_node = self.graph.personal_nodes[self.tracker.state.active_worker]
                self.pending = (action, q_node, action.content)
                return state.probe(f"{q_node}:{action.content}")

            outcome = self.tracker.finish_step(action, None, (None, None))
            if outcome is not Outcome.CONTINUE:
                decision = "fraud" if outcome is Outcome.FRAUD else "not fraud"
                state.notes.append(f"The original RL model decided: {decision}.")
                return None
        return None

    def observe(self, state, probe, observation):
        action, q_node, a_node = self.pending
        answered = a_node if observation.score >= 0.5 else -1
        self.tracker.finish_step(action, answered, (q_node, a_node))
