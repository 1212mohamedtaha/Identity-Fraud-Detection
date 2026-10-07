import json
import random

import numpy as np
import pytest

from fraud_detection import config
from fraud_detection.features import static_features
from fraud_detection.graph import load_graphs, preprocess_graph
from fraud_detection.language import LanguageGenerator
from fraud_detection.models import batch_embedding_lookup, mask_softmax
from fraud_detection.policy import to_one_hot

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def raw_graph():
    return load_graphs()[0]


def test_static_features_match_the_precomputed_ones(raw_graph):
    assert np.allclose(static_features(raw_graph), raw_graph["static_feature"])


def test_preprocess_pads_nodes_and_edges(raw_graph):
    graph = preprocess_graph(raw_graph)
    n_nodes, n_edges = len(raw_graph["nodes"]), len(raw_graph["edges"])
    assert graph.static_feature.shape == (1, n_nodes + 1, config.STATIC_FEATURE_SIZE)
    assert graph.edges.shape == (1, n_edges + 1, 3)
    assert graph.node_edges.max() <= n_edges            # padding points at the padded edge
    assert (graph.workers_actions[0][graph.workers_actions[0] >= n_nodes] == n_nodes).all()
    assert graph.feasible_personal_info_nodes.sum() == len(raw_graph["personal_nodes"])


def test_questions_have_the_right_answer_and_not_sure(raw_graph):
    graph = preprocess_graph(raw_graph)
    generator = LanguageGenerator(graph, random.Random(1))
    head = graph.personal_nodes[0]
    tail = graph.answer_nodes_by_worker[0][0]
    for _ in range(20):
        question, choices = generator.generate(head, tail)
        texts = [text for _, text in choices]
        assert "$" not in question and graph.idx2node[str(head)] in question
        assert graph.idx2node[str(tail)] in texts
        assert texts[-1] == config.NOT_SURE_TEXT
        assert len(set(texts)) == len(texts)
        assert [symbol for symbol, _ in choices] == list(config.OPTIONS[:len(choices)])


def test_mask_softmax_ignores_masked_entries():
    probs = mask_softmax(torch.tensor([[1.0, 2.0, 3.0]]), torch.tensor([[1, 0, 1]]), dim=1)
    assert probs[0, 1] == 0
    assert probs.sum() == pytest.approx(1.0, abs=1e-4)


def test_batch_embedding_lookup_indexes_per_batch_item():
    embeddings = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    out = batch_embedding_lookup(embeddings, torch.tensor([[2, 0], [1, 1]]))
    assert torch.equal(out[0, 0], embeddings[0, 2])
    assert torch.equal(out[1, 1], embeddings[1, 1])


def test_to_one_hot_adds_a_trailing_axis():
    out = to_one_hot(np.array([[0, 2]]), 3)
    assert out.shape == (1, 2, 3)
    assert out[0, 1].tolist() == [0, 0, 1]
