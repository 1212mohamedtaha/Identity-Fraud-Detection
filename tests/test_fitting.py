import json
import random

import pytest

from verity.core.belief import irt_pass_rates, irt_rate
from verity.core.dataset import generate_dataset
from verity.core.fitting import fit_item
from verity.packs.cv import CVPack
from verity.packs.cv.tuning import fit_questions, grader_report


def test_fit_item_recovers_a_known_curve():
    rng = random.Random(0)
    true_rates = irt_pass_rates(5, difficulty=2.3, discrimination=1.7)
    observations = []
    for _ in range(4000):
        level = rng.randrange(5)
        observations.append((level, 1.0 if rng.random() < true_rates[level] else 0.0))
    difficulty, discrimination = fit_item(observations, 5)
    assert difficulty == pytest.approx(2.3, abs=0.2)
    assert discrimination in (1.2, 1.7, 2.4)


def test_fit_questions_and_grader_report(tmp_path):
    generate_dataset(CVPack(), 150, 0, tmp_path)
    out = tmp_path / "fitted.json"
    fitted = fit_questions(tmp_path, out)
    assert json.loads(out.read_text()) == fitted
    assert len(fitted["questions"]) == 56
    assert 0.05 < fitted["score_noise"]["llm"] < fitted["score_noise"]["keyword"] < 0.6
    assert 0 <= fitted["fatigue"] <= 0.08 and fitted["person_spread"] >= 0
    assert abs(sum(fitted["gap_prior"].values()) - 1) < 0.01 and fitted["gap_prior"]["0"] > 0.4
    report = {r["grader"]: r for r in grader_report(tmp_path, "test")}
    assert report["mock-llm"]["mae"] < report["keyword"]["mae"]
    assert 0.5 < report["keyword"]["pass_agreement"] <= 1.0


def test_cv_belief_uses_measured_score_noise():
    offline, with_llm = CVPack(), CVPack()
    with_llm.llm = with_llm.mock_llm()
    assert offline.belief_model().score_noise > with_llm.belief_model().score_noise


def test_fatigue_and_person_spread_are_recovered():
    from verity.core.fitting import fit_fatigue, person_spread

    rng = random.Random(1)
    item = (2.0, 1.7)
    people = []
    for _ in range(300):
        offset = rng.gauss(0, 0.4)
        answers = []
        for position in range(20):
            level = rng.randrange(5)
            rate = irt_rate(level + offset - 0.04 * position, *item)
            answers.append((level, position, 1.0 if rng.random() < rate else 0.0, item))
        people.append(answers)
    assert fit_fatigue(people) == pytest.approx(0.04, abs=0.02)
    assert person_spread(people, 0.04) == pytest.approx(0.4, abs=0.15)
