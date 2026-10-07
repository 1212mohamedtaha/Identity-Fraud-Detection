"""Node features fed to the GNN: static ones computed once per graph, dynamic ones per dialogue turn."""
import json
import math
from itertools import chain

import numpy as np

from . import config


def one_hot(idx, size):
    """One-hot list of length ``size``; indices past the end fall into the last bucket."""
    vector = [0] * size
    vector[min(idx, size - 1)] = 1
    return vector


def static_features(graph, se_freqs_bins=None):
    """Features known before the dialogue starts, one row per node of a raw graph.

    Layout: which personal information the node is (4), is it an answer node (1),
    search-engine frequency bucket (10), degree bucket (10).
    """
    if se_freqs_bins is None:
        with open(config.DATA_ROOT / "se_freqs_bins.json", encoding="utf-8") as f:
            se_freqs_bins = json.load(f)

    answer_nodes = set(chain.from_iterable(graph["one_step_nodes"].values()))
    personal_nodes = list(graph["personal_information"].values())
    rows = []
    for node, se_freqs, degree in zip(graph["nodes"], graph["node_se_freqs"], graph["node_degree"]):
        row = [int(node == personal) for personal in personal_nodes]
        row.append(int(node in answer_nodes))
        bucket = next((i for i, upper in enumerate(se_freqs_bins) if math.log(se_freqs + 1) <= upper),
                      len(se_freqs_bins) - 1)
        row.extend(one_hot(bucket, config.SE_FREQS_FIELD))
        row.extend(one_hot(degree, config.DEGREE_FIELD))
        rows.append(row)
    return rows


def dialogue_features(state):
    """Per-node flags describing the dialogue so far; shape (max_node_num, DYNAMIC_FEATURE_SIZE)."""
    matrix = np.zeros((state.max_node_num, config.DYNAMIC_FEATURE_SIZE), dtype=np.float32)
    for node in state.nodes:
        matrix[node] = (
            node in state.explored_nodes,
            node == state.last_turn_q_node,
            node == state.last_turn_a_node,
            node in state.not_explored_nodes,
            node in state.known_nodes,
            node in state.unknown_nodes,
            node in state.not_answered_nodes,
        )
    return matrix
