# Guide: add a domain pack

A domain pack teaches Verity a new kind of claim: "is a real Marvel fan", "speaks
Spanish", "has used Kubernetes". You write four small classes and register them. The
engine, belief model, policies, RL training, API and UI then work without changes.

The complete, runnable example below is `tests/toy_pack.py`. The CV pack
(`verity/packs/cv/`) is the full-size example with LLM support.

## 1. Create the package

```
verity/packs/trivia/
  __init__.py      # the classes below
  data/            # any files your pack needs (optional)
  prompts/         # LLM prompts, if you use an LLM (optional)
```

## 2. Write the building blocks

```python
from verity.core.graph import KnowledgeGraph
from verity.core.interfaces import ClaimExtractor, KnowledgeSource, ProbeGenerator
from verity.core.pack import DomainPack
from verity.core.types import Choice, Claim, Probe

QUESTIONS = {
    "capitals": [("Capital of France?", "Paris"), ("Capital of Japan?", "Tokyo")],
    "math": [("2 + 2?", "4"), ("3 * 3?", "9")],
}


class TriviaClaims(ClaimExtractor):
    """inputs -> claims. Raise ValueError with a friendly message for bad input."""

    def extract(self, inputs):
        topics = inputs.get("topics", "capitals,math").split(",")
        return [Claim(id=t, text=f"Knows {t}") for t in topics if t in QUESTIONS]


class TriviaKnowledge(KnowledgeSource):
    """claims -> knowledge graph (here just one node per claim)."""

    def build(self, claims, inputs):
        graph = KnowledgeGraph()
        for claim in claims:
            graph.add_node(claim.id, claim.text)
        return graph


class TriviaQuestions(ProbeGenerator):
    """graph -> candidate probes. p_true / p_false say how telling each probe is."""

    def generate(self, claims, graph):
        probes = []
        for claim in claims:
            for i, (question, answer) in enumerate(QUESTIONS[claim.id]):
                choices = [Choice("A", answer), Choice("B", "something else"), Choice("C", "Not Sure")]
                probes.append(Probe(id=f"{claim.id}/{i}", claim_id=claim.id, question=question,
                                    choices=choices, answer="A", p_true=0.9, p_false=0.3))
        return probes


class TriviaPack(DomainPack):
    name = "trivia"                 # used in URLs and the CLI
    title = "Trivia check"
    description = "A tiny quiz."
    input_fields = [{"name": "topics", "label": "Topics", "type": "text", "required": False,
                     "placeholder": "capitals,math"}]
    max_questions = 8

    def claim_extractor(self):
        return TriviaClaims()

    def knowledge_source(self):
        return TriviaKnowledge()

    def probe_generator(self):
        return TriviaQuestions()

    def sample_case(self, rng):
        """A simulated person for training/evaluation: (inputs, truth per claim id)."""
        return {}, {"capitals": rng.random() < 0.5, "math": rng.random() < 0.5}
```

Multiple-choice probes are graded by the default `ChoiceAssessor`. For free-text answers,
leave `choices` empty, put a reference answer in `answer`, and override `assessor()`
(see `AnswerGrader` in the CV pack).

## 3. Register it

In `verity/packs/__init__.py`:

```python
from .trivia import TriviaPack

PACKS = {
    IdentityPack.name: IdentityPack,
    CVPack.name: CVPack,
    TriviaPack.name: TriviaPack,
}
```

## 4. Try it

```bash
verity packs
verity play trivia
verity evaluate trivia
verity train trivia
verity serve            # the pack appears on the start screen
```

## 5. Test it

Add `tests/test_trivia_pack.py`: claims come out right, a person who knows everything is
`supported`, one who knows nothing is `refuted`, bad input raises `ValueError`.
Copy the style of `tests/test_core.py` and `tests/test_cv_pack.py`.

## Choosing `p_true` and `p_false`

They are the most important numbers in a pack.
- `p_true`: how often someone for whom the claim is **true** passes this question
  (0.8–0.9 for fair questions; lower for hard ones).
- `p_false`: how often someone for whom it is **false** still passes: guessing (1 / number
  of real options), or looking it up.
- The bigger the gap, the more one answer counts. Start with sensible guesses; replace them
  with measured rates once you have real data.

## Using an LLM in your pack

Take `self.llm` (it is `None` offline) and pass it to your blocks. Put prompts in
`prompts/<name>.v1.md` and call `ask_json(self.llm, path, **values)`. Always keep an
offline fallback and catch `LLMError`. See the CV pack and [llm-providers.md](llm-providers.md).
