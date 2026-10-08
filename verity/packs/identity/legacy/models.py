"""Neural modules of the hierarchical RL agent: GNN, state trackers, manager and workers.

Attribute names (``linear_cells``, ``transfer``, ``policy_network(s)``, ``fraud_embed`` ...)
must stay as they are: they are the keys of the pretrained checkpoints.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config
from .config import EPS


def batch_embedding_lookup(embeddings, indices):
    """Look up from a batch of embedding matrices.

    :param embeddings: (batch_size, num_items, embedding_size)
    :param indices: (batch_size, num_indices)
    :return: (batch_size, num_indices, embedding_size)
    """
    batch_size, num_items, embed_size = embeddings.shape
    offset = (torch.arange(batch_size, device=indices.device) * num_items).view(batch_size, 1)
    flat_embeddings = embeddings.reshape(-1, embed_size)
    flat_indices = (indices + offset.to(indices.dtype)).reshape(-1)
    return F.embedding(flat_indices, flat_embeddings).reshape(batch_size, -1, embed_size)


def mask_softmax(logits, mask, dim):
    """Softmax over the entries where ``mask`` is non-zero (masked entries get probability 0)."""
    masked_exps = torch.exp(logits) * mask.to(logits.dtype)
    return masked_exps / (masked_exps.sum(dim, keepdim=True) + EPS)


def aggregate(values, dim, method):
    if method == "sum":
        return values.sum(dim)
    if method == "avg":
        return values.sum(dim) / values.shape[dim]
    if method == "max":
        return values.max(dim)[0]
    raise ValueError(f"Unknown message aggregation method: {method}")


class Attn(nn.Module):
    """Scores encoder outputs against a decoder state (dotted / general / concat attention)."""

    def __init__(self, method, encode_hidden_size, decode_hidden_size):
        super().__init__()
        method = method.lower()
        if method not in ("dotted", "general", "concat"):
            raise ValueError(f"Attention method must be dotted, general or concat, got {method!r}")
        if method == "dotted" and encode_hidden_size != decode_hidden_size:
            raise ValueError("Dotted attention needs equal encode and decode hidden sizes")

        self.method = method
        if method == "general":
            self.attn = nn.Linear(encode_hidden_size, decode_hidden_size)
        elif method == "concat":
            joint = encode_hidden_size + decode_hidden_size
            self.attn = nn.Sequential(nn.Linear(joint, joint // 2), nn.Tanh(), nn.Linear(joint // 2, 1))

    def forward(self, encode_outputs, decode_state):
        """
        :param encode_outputs: (batch, output_length, encode_hidden_size)
        :param decode_state: (batch, decode_hidden_size)
        :return: energy (batch, output_length)
        """
        state = decode_state.unsqueeze(1)
        if self.method == "concat":
            state = state.expand(-1, encode_outputs.size(1), -1)
            return self.attn(torch.cat([encode_outputs, state], 2)).squeeze(-1)
        if self.method == "general":
            return torch.sum(state * self.attn(encode_outputs), 2)
        return torch.sum(state * encode_outputs, 2)


class GNN(nn.Module):
    """Message passing network over the knowledge graph."""

    def __init__(self, node_emb_size_list, msg_agg):
        super().__init__()
        self.mp_iters = len(node_emb_size_list) - 1
        self.linear_cells = nn.ModuleList(
            nn.Linear(n_in, n_out) for n_in, n_out in zip(node_emb_size_list[:-1], node_emb_size_list[1:])
        )
        self.msg_agg = msg_agg

    def embed_edges(self, node_embedding, edges, iter_idx):
        """Embed every edge from its sender node. edges: (batch, max_edge_num, 3) = (sender, relation, receiver)."""
        sender_embeds = batch_embedding_lookup(node_embedding, edges[:, :, 0])
        return torch.tanh(self.linear_cells[iter_idx](sender_embeds))

    def pass_message(self, node_edges, node_edge_mask, edge_embeds):
        """Aggregate the embeddings of each node's neighbouring edges.

        :param node_edges: (batch, max_node_num, max_node_edge_num) row ids into ``edge_embeds``
        :param node_edge_mask: same shape as ``node_edges``
        :param edge_embeds: (batch, max_edge_num, feature_size)
        """
        batch_size, node_num, _ = node_edges.shape
        embeds = batch_embedding_lookup(edge_embeds, node_edges.reshape(batch_size, -1))
        embeds = embeds.reshape(batch_size, node_num, -1, edge_embeds.shape[-1])
        embeds = embeds * node_edge_mask.unsqueeze(3).to(embeds.dtype)

        if self.msg_agg == "avg":
            num_neighbors = node_edge_mask.to(torch.float32).sum(2, keepdim=True) + EPS
            return embeds.sum(2) / num_neighbors
        return aggregate(embeds, 2, self.msg_agg)

    def forward(self, initial_node_embed, edges, node_edges, node_edge_mask):
        """:return: (batch, max_node_num, sum(node_emb_size_list))"""
        node_embeds = [initial_node_embed]
        for iter_idx in range(self.mp_iters):
            edge_embeds = self.embed_edges(node_embeds[-1], edges, iter_idx)
            node_embeds.append(self.pass_message(node_edges, node_edge_mask, edge_embeds))
        return torch.cat(node_embeds, 2)


class WorkersStateTracker(nn.Module):
    """Concatenates personal node embeddings with hand-crafted features: the workers' dialogue state."""

    def forward(self, known_one_hot, unknown_one_hot, known_differ_one_hot, workers_qa_turn_one_hot,
                workers_max_qa_turn_one_hot, personal_nodes, final_node_embed):
        """:return: (batch, personal_node_num, feature_size)"""
        personal_node_embed = batch_embedding_lookup(final_node_embed, personal_nodes)
        return torch.cat((known_one_hot, unknown_one_hot, known_differ_one_hot,
                          workers_qa_turn_one_hot, workers_max_qa_turn_one_hot,
                          personal_node_embed), 2)


class ManagerStateTracker(nn.Module):
    """Aggregates personal node embeddings and hand-crafted features: the manager's dialogue state."""

    def __init__(self, personal_node_emb_size, manager_agg_size, msg_agg):
        super().__init__()
        self.msg_agg = msg_agg
        self.transfer = nn.Sequential(nn.Linear(personal_node_emb_size, manager_agg_size), nn.Tanh())

    def forward(self, feasible_personal_info_nodes, workers_decision, known_one_hot, unknown_one_hot,
                known_differ_one_hot, total_qa_turn_one_hot, personal_nodes, final_node_embed):
        """:return: (batch, manager_state_size)"""
        batch_size = personal_nodes.shape[0]
        personal_node_embed = batch_embedding_lookup(final_node_embed, personal_nodes)
        agg_state = aggregate(self.transfer(personal_node_embed), 1, self.msg_agg)

        return torch.cat((feasible_personal_info_nodes,
                          workers_decision.reshape(batch_size, -1),
                          known_one_hot.reshape(batch_size, -1),
                          unknown_one_hot.reshape(batch_size, -1),
                          known_differ_one_hot.reshape(batch_size, -1),
                          total_qa_turn_one_hot,
                          agg_state), 1)


class _PolicyHead(nn.Module):
    """Value network plus learned embeddings for the two terminal actions (fraud / non-fraud)."""

    def __init__(self, state_size, action_emb_size):
        super().__init__()
        self.value_network = nn.Sequential(
            nn.Linear(state_size, state_size // 2),
            nn.Tanh(),
            nn.Linear(state_size // 2, 1),
        )
        self.fraud_embed = nn.Parameter(torch.empty(action_emb_size).uniform_(-1, 1))
        self.non_fraud_embed = nn.Parameter(torch.empty(action_emb_size).uniform_(-1, 1))

    def terminal_embeddings(self, *leading_dims):
        """Both terminal embeddings expanded to (*leading_dims, 1, emb)."""
        shape = (*leading_dims, 1, 1)
        return self.fraud_embed.repeat(*shape), self.non_fraud_embed.repeat(*shape)


class Manager(_PolicyHead):
    """Chooses which personal information (worker) to explore next, or a final decision."""

    def __init__(self, score_method, manager_state_size, worker_state_size):
        super().__init__(manager_state_size, worker_state_size)
        self.policy_network = Attn(score_method, worker_state_size, manager_state_size)

    def forward(self, manager_state, workers_state):
        """
        :param manager_state: (batch, manager_state_size)
        :param workers_state: (batch, personal_node_num, worker_state_size)
        :return: values (batch,), logits (batch, personal_node_num + 2)
        """
        values = self.value_network(manager_state).squeeze(-1)
        fraud, non_fraud = self.terminal_embeddings(manager_state.shape[0])
        actions_embedding = torch.cat((workers_state, fraud, non_fraud), dim=1)
        return values, self.policy_network(actions_embedding, manager_state)


class Workers(_PolicyHead):
    """Each worker picks which answer node to ask about for its personal information, or a decision."""

    def __init__(self, score_method, worker_state_size, answer_node_emb_size):
        super().__init__(worker_state_size, answer_node_emb_size)
        self.policy_networks = Attn(score_method, answer_node_emb_size, worker_state_size)

    def forward(self, workers_state, answer_nodes, graph_node_embedding):
        """
        :param workers_state: (batch, personal_node_num, worker_state_size)
        :param answer_nodes: (batch, personal_node_num, answer_node_num)
        :param graph_node_embedding: (batch, node_num, node_feature_size)
        :return: values (batch, personal_node_num), logits (batch, personal_node_num, answer_node_num + 2)
        """
        values = self.value_network(workers_state).squeeze(-1)

        batch_size, personal_node_num, answer_node_num = answer_nodes.shape
        answer_embedding = batch_embedding_lookup(graph_node_embedding, answer_nodes.reshape(batch_size, -1))
        answer_embedding = answer_embedding.reshape(batch_size, personal_node_num, answer_node_num, -1)
        fraud, non_fraud = self.terminal_embeddings(batch_size, personal_node_num)
        actions_embedding = torch.cat((answer_embedding, fraud, non_fraud), dim=2)

        logits = self.policy_networks(
            actions_embedding.reshape(batch_size * personal_node_num, answer_node_num + 2, -1),
            workers_state.reshape(batch_size * personal_node_num, -1),
        )
        return values, logits.reshape(batch_size, personal_node_num, -1)


def build_models(device=config.DEVICE):
    """Instantiate all networks with the architecture the checkpoint was trained with."""
    node_emb_size_list = [config.INIT_NODE_FEATURE_SIZE, *config.GNN_LAYER_SIZES]
    graph_node_emb_size = sum(node_emb_size_list)
    manager_state_size = config.MANAGER_AGG_SIZE + config.MANAGER_STATE_REST
    worker_state_size = graph_node_emb_size + config.WORKERS_STATE_REST
    models = {
        "gnn": GNN(node_emb_size_list, "max"),
        "manager_state_tracker": ManagerStateTracker(graph_node_emb_size, config.MANAGER_AGG_SIZE, "max"),
        "workers_state_tracker": WorkersStateTracker(),
        "manager": Manager("concat", manager_state_size, worker_state_size),
        "workers": Workers("concat", worker_state_size, graph_node_emb_size),
    }
    return {name: model.to(device) for name, model in models.items()}


def load_models(checkpoint_dir=config.CHECKPOINT_DIR, device=config.DEVICE):
    """Build the networks and load the pretrained weights (tensors only, no arbitrary unpickling)."""
    models = build_models(device)
    for name, model in models.items():
        state_dict = torch.load(Path(checkpoint_dir) / f"{name}.pkl", map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
    return models
