"""Use a dataset's answers.jsonl to fit question parameters and to measure graders."""
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from ...core.belief import irt_pass_rates
from ...core.dataset import read_jsonl
from ...core.fitting import fit_item
from . import LEVELS, keyword_score, skill_bank
from .simulate import MockInterviewLLM


def bank_question(question_id):
    skill, index = question_id.split("/")
    return skill_bank()[skill]["questions"][int(index)]


def fit_questions(data_dir, out):
    """Fit (difficulty, discrimination) per question on the train split, scoring each answer
    by its true quality (what a good grader would give).

    Also measures ``score_noise``: how far scores typically land from the fitted expected
    score, once for a good (LLM-like) grader and once for the offline keyword grader.
    Writes ``out``; returns ``{"questions": {...}, "score_noise": {...}}``.
    """
    rows = [r for r in read_jsonl(Path(data_dir) / "answers.jsonl") if r["split"] == "train"]
    answers = defaultdict(list)
    for row in rows:
        answers[row["question_id"]].append((row["level"], row["quality"]))
    questions = {}
    for question_id, observations in sorted(answers.items()):
        difficulty, discrimination = fit_item(observations, len(LEVELS))
        questions[bank_question(question_id)["question"]] = {
            "difficulty": difficulty, "discrimination": discrimination, "answers": len(observations)}

    squares = {"llm": 0.0, "keyword": 0.0}
    for row in rows:
        question = bank_question(row["question_id"])
        fit = questions[question["question"]]
        expected = irt_pass_rates(len(LEVELS), fit["difficulty"], fit["discrimination"])[row["level"]]
        squares["llm"] += (row["quality"] - expected) ** 2
        squares["keyword"] += (keyword_score(row["answer"], question["keywords"])[0] - expected) ** 2
    noise = {name: round(math.sqrt(total / max(len(rows), 1)), 3) for name, total in squares.items()}

    fitted = {"questions": questions, "score_noise": noise}
    Path(out).write_text(json.dumps(fitted, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return fitted


def grader_report(data_dir, split="test"):
    """Compare graders with the true answer quality on one split.

    Returns rows: grader, answers, mean absolute error, bias (mean score - quality) and
    pass agreement (both >= 0.5 or both < 0.5).
    """
    rows = [r for r in read_jsonl(Path(data_dir) / "answers.jsonl") if r["split"] == split]
    mock = MockInterviewLLM(random.Random(0))
    graders = {
        "keyword": lambda r: keyword_score(r["answer"], bank_question(r["question_id"])["keywords"])[0],
        "mock-llm": lambda r: min(max(r["quality"] + mock.rng.gauss(0, mock.noise), 0.0), 1.0),
    }
    report = []
    for name, grade in graders.items():
        errors, biases, agree = [], [], 0
        for r in rows:
            score = grade(r)
            errors.append(abs(score - r["quality"]))
            biases.append(score - r["quality"])
            agree += (score >= 0.5) == (r["quality"] >= 0.5)
        n = max(len(rows), 1)
        report.append({"grader": name, "answers": len(rows), "mae": sum(errors) / n,
                       "bias": sum(biases) / n, "pass_agreement": agree / n})
    return report
