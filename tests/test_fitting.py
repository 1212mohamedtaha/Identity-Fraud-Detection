import json
import random

import pytest

from verity.core.belief import irt_pass_rates
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
    report = {r["grader"]: r for r in grader_report(tmp_path, "test")}
    assert report["mock-llm"]["mae"] < report["keyword"]["mae"]
    assert 0.5 < report["keyword"]["pass_agreement"] <= 1.0


def test_cv_belief_uses_measured_score_noise():
    offline, with_llm = CVPack(), CVPack()
    with_llm.llm = with_llm.mock_llm()
    assert offline.belief_model().score_noise > with_llm.belief_model().score_noise
