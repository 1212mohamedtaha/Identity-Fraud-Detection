"""Use a dataset's answers.jsonl to fit question parameters and to measure graders."""
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from ...core.belief import irt_rate
from ...core.dataset import read_jsonl
from ...core.fitting import fit_fatigue, fit_item, fit_person_offset, person_spread
from . import LEVELS, keyword_score, skill_bank
from .simulate import MockInterviewLLM


def bank_question(question_id):
    skill, index = question_id.split("/")
    return skill_bank()[skill]["questions"][int(index)]


def fit_questions(data_dir, out):
    """Fit the CV pack's model parameters on a dataset's train split. Scores are each
    answer's true quality (what a good grader would give).

    1. Each question's (difficulty, discrimination), ignoring people's differences.
    2. Fatigue (levels lost per question already asked) and each person's offset.
    3. Each question again, now with every answer placed at level + offset - fatigue x position.
    4. Person spread, from two halves of each person's answers (see fitting.person_spread).
    5. Score noise: distance of scores from the expected score, for LLM-like and keyword grading.

    6. Gap prior: how often a person's real level is 0, 1, 2, ... levels below (or above) the
       level their CV claims, from the train split's cases.

    Writes ``out``; returns ``{"questions", "person_spread", "fatigue", "score_noise", "gap_prior"}``.
    """
    rows = [r for r in read_jsonl(Path(data_dir) / "answers.jsonl") if r["split"] == "train"]
    n_levels = len(LEVELS)

    def fit_all(position_of):
        observations = defaultdict(list)
        for row in rows:
            observations[row["question_id"]].append((position_of(row), row["quality"]))
        return {qid: fit_item(obs, n_levels) for qid, obs in sorted(observations.items())}

    def people_answers(items):
        people = defaultdict(list)
        for row in rows:
            people[row["case"]].append((row["level"], row["position"], row["quality"], items[row["question_id"]]))
        return people

    items = fit_all(lambda row: row["level"])                                   # step 1
    people = people_answers(items)
    fatigue = fit_fatigue(list(people.values()))                                # step 2
    offsets = {case: fit_person_offset(answers, fatigue) for case, answers in people.items()}
    items = fit_all(lambda row: row["level"] + offsets[row["case"]] - fatigue * row["position"])  # step 3
    people = people_answers(items)
    spread = person_spread(list(people.values()), fatigue)                      # step 4
    offsets = {case: fit_person_offset(answers, fatigue) for case, answers in people.items()}

    squares = {"llm": 0.0, "keyword": 0.0}                                      # step 5
    for row in rows:
        question = bank_question(row["question_id"])
        x = row["level"] + offsets[row["case"]] - fatigue * row["position"]
        expected = irt_rate(x, *items[row["question_id"]])
        squares["llm"] += (row["quality"] - expected) ** 2
        squares["keyword"] += (keyword_score(row["answer"], question["keywords"])[0] - expected) ** 2
    noise = {name: round(math.sqrt(total / max(len(rows), 1)), 3) for name, total in squares.items()}

    questions = {bank_question(qid)["question"]: {"difficulty": d, "discrimination": a,
                                                  "answers": sum(1 for r in rows if r["question_id"] == qid)}
                 for qid, (d, a) in items.items()}
    gaps = defaultdict(int)                                                     # step 6
    for case in read_jsonl(Path(data_dir) / "cases.jsonl"):
        if case["split"] == "train":
            for skill, real in case["truth"].items():
                gaps[real - case["claimed"][skill]] += 1
    count = sum(gaps.values())
    gap_prior = {str(gap): round(n / count, 4) for gap, n in sorted(gaps.items())}

    fitted = {"questions": questions, "person_spread": round(spread, 3), "fatigue": fatigue,
              "score_noise": noise, "gap_prior": gap_prior}
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
