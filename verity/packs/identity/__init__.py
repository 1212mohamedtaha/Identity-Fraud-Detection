"""Identity pack: verify someone's claimed workplace, university, home and birthplace
by asking about places near them. Built on the original project's data and model.
"""
import random
from functools import lru_cache

from ...core.graph import KnowledgeGraph
from ...core.interfaces import ClaimExtractor, KnowledgeSource, ProbeGenerator
from ...core.pack import DomainPack
from ...core.types import Choice, Claim, Probe
from .legacy.graph import load_graphs, preprocess_graph
from .legacy.language import LanguageGenerator

CLAIM_TEXT = {
    "company": "Works at {}",
    "university": "Studied at {}",
    "live_in": "Lives at {}",
    "born_in": "Was born in {}",
}

# Share of simulated people per number of falsified attributes (0 = genuine),
# taken from the original project's simulator (Non-Fraud 4 : Type-1..4 Fraud 1 each).
FRAUD_TYPE_WEIGHTS = {0: 4, 1: 1, 2: 1, 3: 1, 4: 1}


@lru_cache(maxsize=None)
def graphs():
    return load_graphs()


@lru_cache(maxsize=None)
def processed_graph(index):
    return preprocess_graph(graphs()[index])


class IdentityClaims(ClaimExtractor):
    """One claim per personal attribute stored in the applicant's graph."""

    def extract(self, inputs):
        index = int(inputs.get("graph", 0))
        raw = graphs()[index]
        claims = []
        for kind, node in raw["personal_information"].items():
            name = raw["idx2node"][str(node)]
            text = CLAIM_TEXT.get(kind, kind + ": {}").format(name)
            claims.append(Claim(id=kind, text=text, kind=kind, data={"node": node, "graph": index}))
        return claims


class PlacesGraph(KnowledgeSource):
    """Places around each personal attribute (collected with Google Places in the original project)."""

    def build(self, claims, inputs):
        index = int(inputs.get("graph", 0))
        raw = graphs()[index]
        processed = processed_graph(index)
        graph = KnowledgeGraph(data={"graph_index": index, "processed": processed})
        for idx, name in raw["idx2node"].items():
            graph.add_node(idx, name)
        for claim in claims:
            node = claim.data["node"]
            for answer in raw["one_step_nodes"][str(node)]:
                relation = processed.h_t_to_r[f"{node} {answer}"]
                graph.add_edge(str(node), relation, str(answer), provenance="google-places")
        return graph


class PlaceQuestions(ProbeGenerator):
    """One multiple-choice question per (personal attribute, nearby place) pair."""

    p_true = 0.85    # a genuine person usually knows places near their own home or work
    p_false = 0.30   # an impostor mostly guesses among three places

    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def generate(self, claims, graph):
        processed = graph.data["processed"]
        language = LanguageGenerator(processed, self.rng)
        probes = []
        for claim in claims:
            node = claim.data["node"]
            for answer in processed.one_step_nodes[str(node)]:
                question, options = language.generate(node, answer)
                correct = processed.idx2node[str(answer)]
                correct_id = next(symbol for symbol, text in options if text == correct)
                probes.append(Probe(
                    id=f"{node}:{answer}",
                    claim_id=claim.id,
                    question=question,
                    choices=[Choice(symbol, text) for symbol, text in options],
                    answer=correct_id,
                    p_true=self.p_true,
                    p_false=self.p_false,
                    data={"q_node": node, "a_node": answer},
                ))
        return probes


class IdentityPack(DomainPack):
    name = "identity"
    title = "Identity check"
    description = "Answer questions about places near your workplace, university and home."
    input_fields = []
    max_questions = 20

    def claim_extractor(self):
        return IdentityClaims()

    def knowledge_source(self):
        return PlacesGraph()

    def probe_generator(self):
        return PlaceQuestions()

    def policy_names(self):
        return super().policy_names() + ["legacy"]

    def policy(self, name="greedy", rng=None):
        if name == "legacy":
            from .legacy_policy import LegacyRLPolicy
            return LegacyRLPolicy()
        return super().policy(name, rng)

    def sample_case(self, rng):
        index = rng.randrange(len(graphs()))
        kinds = list(graphs()[index]["personal_information"])
        weights = FRAUD_TYPE_WEIGHTS
        falsified = rng.choices(list(weights), weights=list(weights.values()))[0]
        false_kinds = set(rng.sample(kinds, min(falsified, len(kinds))))
        truth = {kind: kind not in false_kinds for kind in kinds}
        return {"graph": index}, truth
