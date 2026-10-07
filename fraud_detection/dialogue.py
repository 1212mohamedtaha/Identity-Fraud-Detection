"""Dialogue state and the rules for updating it after each system action / user answer."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from . import config
from .config import FRAUD, NON_FRAUD
from .features import dialogue_features
from .graph import ProcessedGraph


class Outcome(Enum):
    CONTINUE = "continue"
    FRAUD = "fraud"
    NON_FRAUD = "non_fraud"


@dataclass(frozen=True)
class SystemAction:
    """One decision of the hierarchical policy.

    ``role`` is ``"manager"`` (``content`` is a personal node to explore, or FRAUD / NON_FRAUD) or
    ``"worker"`` (``content`` is an answer node to ask about, or FRAUD / NON_FRAUD).
    ``index`` is the position of ``content`` in that agent's action space.
    """

    role: str
    index: int
    content: int

    @property
    def is_terminal(self):
        return self.content in (FRAUD, NON_FRAUD)


@dataclass
class DialogueState:
    # Constants of the graph.
    nodes: list
    personal_nodes: list
    max_node_num: int

    # Nodes seen so far (the "dialogue feature" of the GNN).
    explored_nodes: set
    not_explored_nodes: set
    known_nodes: set
    unknown_nodes: set
    not_answered_nodes: set
    last_turn_q_node: Optional[int] = None
    last_turn_a_node: Optional[int] = None
    dialogue_feature: Optional[np.ndarray] = None

    # Turn counters.
    total_exploring_turn: int = 0
    current_worker_exploring_turn: int = 0
    total_qa_turn: int = 0
    workers_qa_turn: np.ndarray = None
    known_counts: np.ndarray = None      # correct answers per worker
    unknown_counts: np.ndarray = None    # wrong / "not sure" answers per worker

    # Who acts: one-hot over [manager, worker_0, worker_1, ...].
    policy_mask: np.ndarray = None
    # Which actions are still allowed; the last two entries are [FRAUD, NON_FRAUD].
    manager_action_mask: np.ndarray = None
    workers_action_mask: np.ndarray = None

    def refresh_dialogue_feature(self):
        self.dialogue_feature = dialogue_features(self)

    @property
    def active_worker(self):
        """Index of the worker currently exploring, or None while the manager acts."""
        policy_idx = int(self.policy_mask.argmax())
        return policy_idx - 1 if policy_idx else None


def initial_state(graph: ProcessedGraph) -> DialogueState:
    num_workers = len(graph.personal_nodes)
    answers_per_worker = graph.answer_nodes_by_worker
    max_answers = graph.workers_actions.shape[2]

    manager_mask = np.zeros(num_workers + 2, dtype=np.int32)
    manager_mask[:num_workers] = graph.feasible_personal_info_nodes[0]

    workers_mask = np.zeros((num_workers, max_answers + 2), dtype=np.int32)
    for worker_idx, answers in enumerate(answers_per_worker):
        workers_mask[worker_idx, :len(answers)] = 1

    state = DialogueState(
        nodes=list(graph.nodes),
        personal_nodes=list(graph.personal_nodes),
        max_node_num=graph.max_node_num,
        explored_nodes=set(),
        not_explored_nodes=set(graph.nodes),
        known_nodes=set(),
        unknown_nodes=set(),
        not_answered_nodes={node for answers in answers_per_worker for node in answers},
        workers_qa_turn=np.zeros(num_workers, dtype=np.int32),
        known_counts=np.zeros(num_workers, dtype=np.int32),
        unknown_counts=np.zeros(num_workers, dtype=np.int32),
        policy_mask=np.asarray([1] + [0] * num_workers, dtype=np.int32),
        manager_action_mask=manager_mask,
        workers_action_mask=workers_mask,
    )
    state.refresh_dialogue_feature()
    return state


class DialogueTracker:
    """Applies system actions and user answers to a :class:`DialogueState`."""

    def __init__(self, graph: ProcessedGraph):
        self.state = initial_state(graph)

    def begin_step(self, action: SystemAction):
        """Register the chosen action and force termination once the turn budget is spent."""
        state = self.state
        state.total_exploring_turn += 1
        if state.total_exploring_turn >= config.MAX_EXPLORING_TIME_STEP:
            state.manager_action_mask[:-2] = 0
            state.manager_action_mask[-2:] = 1
            state.workers_action_mask[:, :-2] = 0
            state.workers_action_mask[:, -2:] = 1

    def finish_step(self, action: SystemAction, answered_node: Optional[int], question_nodes) -> Outcome:
        """Update the state after ``action`` was carried out.

        :param answered_node: node the user picked as the answer, ``-1`` for "not sure" / unknown option,
            or ``None`` when no question was asked this step.
        :param question_nodes: ``(q_node, a_node)`` of the question that was asked, or ``(None, None)``.
        """
        state = self.state
        q_node, a_node = question_nodes

        if answered_node is not None:
            state.total_qa_turn += 1
        state.last_turn_q_node = None
        state.last_turn_a_node = None

        if action.role == "manager":
            if action.is_terminal:
                return Outcome.FRAUD if action.content == FRAUD else Outcome.NON_FRAUD
            self._start_exploring(action)
        elif action.is_terminal:
            self._finish_worker(action)
        else:
            self._record_answer(action, answered_node, q_node, a_node)

        state.refresh_dialogue_feature()
        return Outcome.CONTINUE

    def _start_exploring(self, action):
        """The manager picked a personal node: hand control to the worker of that node."""
        state = self.state
        state.current_worker_exploring_turn = 0
        state.explored_nodes.add(action.content)
        state.not_explored_nodes.discard(action.content)
        state.manager_action_mask[action.index] = 0

        worker_idx = state.personal_nodes.index(action.content)
        state.policy_mask = np.zeros(len(state.personal_nodes) + 1, dtype=np.int32)
        state.policy_mask[worker_idx + 1] = 1

    def _finish_worker(self, action):
        """A worker made its decision: control goes back to the manager."""
        state = self.state
        if action.content == FRAUD or state.manager_action_mask.sum() == 0:
            state.manager_action_mask[-2:] = 1
        state.policy_mask = np.zeros(len(state.personal_nodes) + 1, dtype=np.int32)
        state.policy_mask[0] = 1

    def _record_answer(self, action, answered_node, q_node, a_node):
        """A worker asked about ``action.content`` and the user answered."""
        state = self.state
        worker_idx = state.active_worker
        state.current_worker_exploring_turn += 1
        state.workers_qa_turn[worker_idx] += 1

        state.last_turn_q_node = q_node
        state.last_turn_a_node = a_node
        state.explored_nodes.add(action.content)
        state.not_explored_nodes.discard(action.content)
        state.not_answered_nodes.discard(action.content)
        if answered_node == a_node:
            state.known_nodes.add(action.content)
            state.known_counts[worker_idx] += 1
        else:
            state.unknown_nodes.add(action.content)
            state.unknown_counts[worker_idx] += 1

        state.workers_action_mask[worker_idx, action.index] = 0
        if (state.workers_action_mask[worker_idx].sum() == 0
                or state.workers_qa_turn[worker_idx] >= config.MIN_WORKER_QA_TURN):
            state.workers_action_mask[worker_idx, -2:] = 1
        if state.current_worker_exploring_turn >= config.MAX_WORKER_EXPLORING_TIME_STEP:
            # The worker has to decide on its next turn.
            state.workers_action_mask[worker_idx, :-2] = 0
