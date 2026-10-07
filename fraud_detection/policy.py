"""Runs the trained hierarchical policy on a dialogue state."""
import numpy as np
import torch

from . import config
from .config import FRAUD, NON_FRAUD
from .dialogue import DialogueState, SystemAction
from .graph import ProcessedGraph
from .models import load_models, mask_softmax


def to_one_hot(values, size):
    """One-hot encode an integer array of any shape along a new last axis."""
    values = np.asarray(values)
    return np.eye(size, dtype=np.float32)[values]


def _tensor(array, dtype=torch.float32):
    return torch.as_tensor(np.asarray(array), dtype=dtype, device=config.DEVICE)


class Policy:
    """Greedy (arg-max) hierarchical policy: a manager picks what to explore, workers pick questions."""

    def __init__(self, models=None):
        self.models = models if models is not None else load_models()

    @torch.no_grad()
    def decide(self, graph: ProcessedGraph, state: DialogueState) -> SystemAction:
        """Pick the next action of whichever agent (manager or one of the workers) is in control."""
        models = self.models
        steps = config.MAX_WORKER_EXPLORING_TIME_STEP + 1

        node_features = np.concatenate([graph.static_feature, state.dialogue_feature[None]], axis=-1)
        final_node_embed = models["gnn"](
            _tensor(node_features),
            _tensor(graph.edges, torch.long),
            _tensor(graph.node_edges, torch.long),
            _tensor(graph.node_edge_mask, torch.uint8),
        )

        known = state.known_counts[None]
        unknown = state.unknown_counts[None]
        known_differ = np.maximum(known - unknown, 0)
        personal_nodes = _tensor(graph.manager_actions, torch.long)

        known_one_hot = _tensor(to_one_hot(known, steps))
        unknown_one_hot = _tensor(to_one_hot(unknown, steps))
        known_differ_one_hot = _tensor(to_one_hot(known_differ, steps))

        workers_state = models["workers_state_tracker"](
            known_one_hot,
            unknown_one_hot,
            known_differ_one_hot,
            _tensor(to_one_hot(state.workers_qa_turn[None], steps)),
            _tensor(to_one_hot(graph.workers_max_qa_turn, steps)),
            personal_nodes,
            final_node_embed,
        )
        _, workers_logits = models["workers"](
            workers_state, _tensor(graph.workers_actions, torch.long), final_node_embed)
        workers_probs = mask_softmax(workers_logits, _tensor(state.workers_action_mask[None], torch.uint8), dim=2)

        manager_state = models["manager_state_tracker"](
            _tensor(graph.feasible_personal_info_nodes),
            workers_probs[:, :, -2:],     # each worker's current (fraud, non-fraud) leaning
            known_one_hot,
            unknown_one_hot,
            known_differ_one_hot,
            _tensor(to_one_hot(np.array([state.total_qa_turn]), config.MAX_EXPLORING_TIME_STEP + 1)),
            personal_nodes,
            final_node_embed,
        )
        _, manager_logits = models["manager"](manager_state, workers_state)
        manager_probs = mask_softmax(manager_logits, _tensor(state.manager_action_mask[None], torch.uint8), dim=1)

        terminal = [FRAUD, NON_FRAUD]
        worker_idx = state.active_worker
        if worker_idx is None:
            index = int(manager_probs[0].argmax())
            content = (list(graph.manager_actions[0]) + terminal)[index]
            return SystemAction("manager", index, int(content))

        index = int(workers_probs[0, worker_idx].argmax())
        content = (list(graph.workers_actions[0, worker_idx]) + terminal)[index]
        return SystemAction("worker", index, int(content))
