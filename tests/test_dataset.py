import json
import random

from verity.core.dataset import generate_dataset, load_cases, split_for
from verity.core.engine import Session
from verity.core.simulation import evaluate, run_episode
from verity.packs.cv import CVPack, simulate
from verity.packs.cv.simulate import MockInterviewLLM, PersonaRespondent, make_persona, write_answer


def test_splits_are_70_15_15():
    splits = [split_for(i, 100) for i in range(100)]
    assert splits.count("train") == 70 and splits.count("val") == 15 and splits.count("test") == 15


def test_generated_dataset_files(tmp_path, toy):
    cases = generate_dataset(toy, 20, 0, tmp_path)
    assert len(load_cases(tmp_path)) == 20 and len(load_cases(tmp_path, "test")) == 3
    assert json.loads((tmp_path / "info.json").read_text())["pack"] == "toy"
    assert cases == generate_dataset(toy, 20, 0, tmp_path / "again")        # same seed, same data


def test_cv_dataset_has_personas_answers_and_hidden_difficulty(tmp_path):
    cases = generate_dataset(CVPack(), 30, 1, tmp_path)
    persona = cases[0]
    assert set(persona) >= {"id", "split", "inputs", "truth", "claimed", "honesty", "traits"}
    claims = Session(CVPack(), persona["inputs"]).state.claims
    assert {c.id: c.claimed_level for c in claims} == persona["claimed"]       # the CV says what was claimed
    rows = [json.loads(line) for line in (tmp_path / "answers.jsonl").read_text().splitlines()]
    assert rows and all(0 <= r["quality"] <= 1 for r in rows)
    questions = json.loads((tmp_path / "questions.json").read_text())
    assert all(abs(q["hidden_difficulty"] - {"easy": 1.5, "medium": 2.5, "hard": 3.5}[q["label"]]) <= 0.75
               for q in questions.values())


def test_impostors_claim_high_and_know_little():
    rng = random.Random(0)
    impostors = [p for p in (make_persona(rng, i) for i in range(300)) if p["honesty"] == "impostor"]
    assert impostors
    for p in impostors:
        assert all(p["truth"][k] <= 1 < p["claimed"][k] for k in p["truth"])


def test_answers_get_better_with_quality_and_hidden_difficulty_is_stable():
    rng = random.Random(0)
    points = ["alpha", "beta", "gamma", "delta"]
    assert write_answer(points, 0.05, rng) in ("I'm not sure.", "I don't know, sorry.",
                                               "No idea, I haven't used that much.")
    assert sum(p in write_answer(points, 1.0, rng) for p in points) >= 2
    assert simulate.hidden_difficulty("q", "easy") == simulate.hidden_difficulty("q", "easy")


def test_mock_llm_runs_the_whole_llm_path():
    pack = CVPack()
    pack.llm = pack.mock_llm(seed=0)
    persona = make_persona(random.Random(3), 0)
    session = Session(pack, persona["inputs"])
    assert {c.id for c in session.state.claims} == set(persona["truth"])
    assert all("/llm-" in p.id for p in session.state.probes)            # questions came through the LLM path
    respondent = PersonaRespondent(persona, random.Random(0), pack.llm.quality_index)
    answer = respondent.answer(session.current)
    observation = session.answer(answer)
    assert observation.feedback == "Mock grade."
    assert abs(observation.score - pack.llm.quality_index[answer]) < 0.4


def test_simulation_over_dataset_cases(tmp_path):
    pack = CVPack()
    cases = generate_dataset(pack, 40, 2, tmp_path)
    session, truth = run_episode(pack, pack.policy("greedy"), random.Random(0), cases[0])
    assert session.finished and set(truth) == set(cases[0]["truth"])
    result = evaluate(pack, "greedy", cases=load_cases(tmp_path, "test"))
    assert result["episodes"] == 6


def test_mock_llm_grades_unknown_answers_by_key_points():
    llm = MockInterviewLLM(random.Random(0), noise=0)
    prompt = "Key points: alpha; beta\n\n<answer>\nalpha and beta\n</answer>"
    reply = llm.complete("You grade interview answers fairly", prompt)
    assert json.loads(reply)["score"] == 1.0
