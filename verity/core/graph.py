"""A tiny knowledge graph: nodes, labelled edges and where each fact came from."""
from dataclasses import dataclass, field


@dataclass
class Edge:
    source: str
    relation: str
    target: str
    provenance: str = ""     # where this fact came from (file, URL, "llm", ...)


@dataclass
class KnowledgeGraph:
    nodes: dict = field(default_factory=dict)   # node id -> {"label": ..., any other attributes}
    edges: list = field(default_factory=list)   # list[Edge]
    data: dict = field(default_factory=dict)    # pack-specific extras

    def add_node(self, node_id, label, **attributes):
        self.nodes[node_id] = {"label": label, **attributes}

    def add_edge(self, source, relation, target, provenance=""):
        self.edges.append(Edge(source, relation, target, provenance))

    def label(self, node_id):
        return self.nodes[node_id]["label"]

    def neighbours(self, node_id, relation=None):
        return [e.target for e in self.edges
                if e.source == node_id and (relation is None or e.relation == relation)]

    def to_dict(self):
        return {
            "nodes": self.nodes,
            "edges": [vars(edge) for edge in self.edges],
        }
