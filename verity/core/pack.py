"""DomainPack: the one class a use case extends to plug into the engine.

Override the factory methods for the parts your domain does differently; keep the
defaults for the rest. See docs/guides/adding-a-pack.md.
"""
import random
from pathlib import Path

from . import policies
from .belief import BeliefModel
from .interfaces import ChoiceAssessor
from .simulation import StatisticalRespondent

MODELS_DIR = Path("models")      # where trained policies are saved, relative to the working dir


class DomainPack:
    name = "pack"                 # short id used in URLs and the CLI
    title = "Pack"                # human-friendly name
    description = ""
    input_fields = []             # form fields for the UI: dicts with name, label, type, required
    max_questions = 10            # hard limit per session
    show_feedback = False         # show graded feedback to the user after each answer
    prior = 0.5                   # starting belief that a claim is true
    accept = 0.9                  # belief needed to mark a claim supported
    reject = 0.1                  # belief at which a claim is marked refuted

    def __init__(self, llm=None):
        self.llm = llm            # an LLM from verity.llm, or None to run fully offline

    # ----- building blocks: subclasses must provide these three -----
    def claim_extractor(self):
        raise NotImplementedError

    def knowledge_source(self):
        raise NotImplementedError

    def probe_generator(self):
        raise NotImplementedError

    # ----- building blocks with defaults -----
    def assessor(self):
        return ChoiceAssessor()

    def belief_model(self):
        return BeliefModel(prior=self.prior, accept=self.accept, reject=self.reject)

    def policy_names(self):
        names = ["greedy", "random"]
        if self.learned_policy_path().exists():
            names.append("learned")
        return names

    def policy(self, name="greedy", rng=None):
        if name == "greedy":
            return policies.GreedyPolicy()
        if name == "random":
            return policies.RandomPolicy(rng)
        if name == "learned":
            from ..rl.policy import load_learned_policy
            return load_learned_policy(self.learned_policy_path())
        raise ValueError(f"Unknown policy {name!r} for pack {self.name!r}; "
                         f"choose from {self.policy_names()}")

    def learned_policy_path(self):
        return MODELS_DIR / self.name / "policy.pt"

    # ----- simulation (training and evaluation) -----
    def sample_case(self, rng):
        """Return (inputs, truth) for one simulated person.

        ``truth`` maps claim id -> the person's real level (an index into the claim's
        levels), or for yes/no claims simply True / False.
        """
        raise NotImplementedError

    def make_case(self, rng, index):
        """One simulated person as a dataset row. Override to add traits (see the CV pack)."""
        inputs, truth = self.sample_case(rng)
        return {"id": f"{self.name}-{index:05d}", "inputs": inputs, "truth": truth}

    def write_dataset_extras(self, cases, out, rng):
        """Hook to write extra dataset files next to cases.jsonl (default: none)."""

    def respondent(self, truth, rng=None, case=None):
        """The simulated person answering. ``truth`` maps claim id -> real level."""
        return StatisticalRespondent(truth, rng or random.Random())

    def mock_llm(self, seed=0):
        """A fake LLM that plays this pack's prompts, for simulations; None if the pack has none."""
        return None

    def describe(self):
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "fields": self.input_fields,
            "policies": self.policy_names(),
        }
