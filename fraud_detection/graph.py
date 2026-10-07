"""Turns a raw knowledge graph (one applicant) into the padded arrays consumed by the networks."""
import json
from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class ProcessedGraph:
    """Network inputs for one applicant graph, all with a leading batch dimension of 1."""

    nodes: list
    personal_nodes: list
    one_step_nodes: dict
    idx2node: dict
    h_t_to_r: dict           # "head tail" -> relation id, used to phrase questions

    max_node_num: int
    static_feature: np.ndarray        # (1, max_node_num, STATIC_FEATURE_SIZE)
    edges: np.ndarray                 # (1, max_edge_num, 3)
    node_edges: np.ndarray            # (1, max_node_num, max_node_edge_num)
    node_edge_mask: np.ndarray        # (1, max_node_num, max_node_edge_num)
    feasible_personal_info_nodes: np.ndarray   # (1, personal_node_num)
    manager_actions: np.ndarray       # (1, personal_node_num)
    workers_actions: np.ndarray       # (1, personal_node_num, max_answer_node_num)
    workers_max_qa_turn: np.ndarray   # (1, personal_node_num)

    @property
    def answer_nodes_by_worker(self):
        """Answer nodes available to each personal node, in worker order."""
        return [self.one_step_nodes[str(node)] for node in self.personal_nodes]


def load_graphs(path=config.GRAPHS_FILE):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def preprocess_graph(graph):
    nodes = list(graph["nodes"])
    personal_nodes = list(graph["personal_nodes"])
    one_step_nodes = graph["one_step_nodes"]
    num_nodes, num_edges = len(nodes), len(graph["edges"])

    # One extra row each: the padded node / edge index is the real count.
    max_node_num = num_nodes + 1
    max_edge_num = num_edges + 1
    max_node_edge_num = max(map(len, graph["node_edges"]))
    max_answer_node_num = max(map(len, one_step_nodes.values()))

    static_feature = np.zeros((max_node_num, config.STATIC_FEATURE_SIZE), dtype=np.float32)
    static_feature[:num_nodes] = np.asarray(graph["static_feature"], dtype=np.float32)

    edges = np.zeros((max_edge_num, 3), dtype=np.int32)
    edges[:num_edges] = np.asarray(graph["edges"], dtype=np.int32)
    edges[num_edges:] = config.EDGE_PAD

    node_edges = np.full((max_node_num, max_node_edge_num), num_edges, dtype=np.int32)
    node_edge_mask = np.zeros((max_node_num, max_node_edge_num), dtype=np.int32)
    for i, incident in enumerate(graph["node_edges"]):
        node_edges[i, :len(incident)] = incident
        node_edge_mask[i, :len(incident)] = 1

    feasible = np.zeros(len(personal_nodes), dtype=np.float32)
    workers_actions = np.full((len(personal_nodes), max_answer_node_num), num_nodes, dtype=np.int32)
    workers_max_qa_turn = np.zeros(len(personal_nodes), dtype=np.int32)
    for worker_idx, personal_node in enumerate(personal_nodes):
        answers = one_step_nodes[str(personal_node)]
        if answers:
            feasible[worker_idx] = 1
            workers_actions[worker_idx, :len(answers)] = answers
            workers_max_qa_turn[worker_idx] = min(len(answers), config.MAX_WORKER_EXPLORING_TIME_STEP)

    h_t_to_r = {f"{receiver} {sender}": str(relation) for sender, relation, receiver in graph["edges"]}

    return ProcessedGraph(
        nodes=nodes,
        personal_nodes=personal_nodes,
        one_step_nodes=one_step_nodes,
        idx2node=dict(graph["idx2node"]),
        h_t_to_r=h_t_to_r,
        max_node_num=max_node_num,
        static_feature=static_feature[None],
        edges=edges[None],
        node_edges=node_edges[None],
        node_edge_mask=node_edge_mask[None],
        feasible_personal_info_nodes=feasible[None],
        manager_actions=np.asarray(personal_nodes, dtype=np.int32)[None],
        workers_actions=workers_actions[None],
        workers_max_qa_turn=workers_max_qa_turn[None],
    )
