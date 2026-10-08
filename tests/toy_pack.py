"""The smallest possible pack: two claims, multiple-choice probes, no files, no LLM.

Also used as the worked example in docs/guides/adding-a-pack.md.
"""
from verity.core.graph import KnowledgeGraph
from verity.core.interfaces import ClaimExtractor, KnowledgeSource, ProbeGenerator
from verity.core.pack import DomainPack
from verity.core.types import Choice, Claim, Probe

QUESTIONS = {
    "capitals": [("Capital of France?", "Paris"), ("Capital of Japan?", "Tokyo"),
                 ("Capital of Egypt?", "Cairo"), ("Capital of Peru?", "Lima")],
    "math": [("2 + 2?", "4"), ("3 * 3?", "9"), ("10 / 2?", "5"), ("7 - 4?", "3")],
}


class ToyClaims(ClaimExtractor):
    def extract(self, inputs):
        topics = inputs.get("topics", "capitals,math").split(",")
        return [Claim(id=t, text=f"Knows {t}") for t in topics if t in QUESTIONS]


class ToyKnowledge(KnowledgeSource):
    def build(self, claims, inputs):
        graph = KnowledgeGraph()
        for claim in claims:
            graph.add_node(claim.id, claim.text)
        return graph


class ToyQuestions(ProbeGenerator):
    def generate(self, claims, graph):
        probes = []
        for claim in claims:
            for i, (question, answer) in enumerate(QUESTIONS[claim.id]):
                choices = [Choice("A", answer), Choice("B", "something else"), Choice("C", "Not Sure")]
                probes.append(Probe(id=f"{claim.id}/{i}", claim_id=claim.id, question=question,
                                    choices=choices, answer="A", p_true=0.9, p_false=0.3))
        return probes


class ToyPack(DomainPack):
    name = "toy"
    title = "Toy quiz"
    max_questions = 8

    def claim_extractor(self):
        return ToyClaims()

    def knowledge_source(self):
        return ToyKnowledge()

    def probe_generator(self):
        return ToyQuestions()

    def sample_case(self, rng):
        return {}, {"capitals": rng.random() < 0.5, "math": rng.random() < 0.5}
