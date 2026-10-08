"""Turns graph triples into natural-language multiple-choice questions."""
import json
import random
import re
from functools import lru_cache

from . import config


@lru_cache(maxsize=None)
def _load(name):
    with open(config.DATA_ROOT / name, encoding="utf-8") as f:
        return json.load(f)


class LanguageGenerator:
    def __init__(self, graph, rng=random):
        self.graph = graph
        self.rng = rng
        self.idx2r = _load("idx2r.json")
        self.answers_library = _load("answersLibrary.json")
        self.templates = _load("languageTemplates.json")

    def node_name(self, node):
        return self.graph.idx2node[str(node)]

    def generate(self, head, tail):
        """Question about ``head`` whose correct answer is ``tail``.

        :return: (question, [(option_symbol, answer_text), ...]); the last option is always "Not Sure".
        """
        relation = self.idx2r[self.graph.h_t_to_r[f"{head} {tail}"]]
        head_name, answer = self.node_name(head), self.node_name(tail)

        question = self.rng.choice(self.templates[relation])
        question = re.sub(r"\$\S\$", lambda _: head_name, question)

        # Distractors must look like the real answer, but must not contain it or be contained in it.
        distractors = [c for c in self.answers_library[relation]
                       if c != answer and answer not in c and c not in answer]
        options = self.rng.sample(distractors, min(config.NEGATIVE_SAMPLED_ANSWER_NUM, len(distractors)))
        options.append(answer)
        self.rng.shuffle(options)
        options.append(config.NOT_SURE_TEXT)
        return question, list(zip(config.OPTIONS, options))
