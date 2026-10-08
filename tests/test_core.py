import random

import pytest

from verity.core.belief import BeliefModel, irt_pass_rates
from verity.core.engine import Session, SessionError
from verity.core.policies import GreedyPolicy
from verity.core.simulation import StatisticalRespondent, evaluate, true_levels
from verity.core.types import REFUTED, SUPPORTED, Claim, Observation, Probe


def probe(claim_id="c", p_true=0.9, p_false=0.3):
    return Probe(id="p", claim_id=claim_id, question="?", p_true=p_true, p_false=p_false)


def test_belief_moves_up_on_pass_and_down_on_fail():
    belief = BeliefModel()
    belief.start([Claim("c", "claim")])
    belief.update(probe(), Observation("p", "c", "x", 1.0))
    assert belief.probability("c") > 0.5
    belief.start([Claim("c", "claim")])
    belief.update(probe(), Observation("p", "c", "x", 0.0))
    assert belief.probability("c") < 0.5


def test_partial_score_moves_less_than_full_pass():
    full, half = BeliefModel(), BeliefModel()
    for b, score in ((full, 1.0), (half, 0.6)):
        b.start([Claim("c", "claim")])
        b.update(probe(), Observation("p", "c", "x", score))
    assert 0.5 < half.probability("c") < full.probability("c")


def test_statuses_follow_thresholds():
    belief = BeliefModel(accept=0.9, reject=0.1)
    belief.start([Claim("c", "claim")])
    for _ in range(5):
        belief.update(probe(), Observation("p", "c", "x", 1.0))
    assert belief.status("c") == SUPPORTED


def test_more_discriminating_probes_are_worth_more():
    belief = BeliefModel()
    belief.start([Claim("c", "claim")])
    assert belief.expected_gain(probe(p_true=0.95, p_false=0.05)) > belief.expected_gain(probe(p_true=0.6, p_false=0.4))


def test_session_supports_a_person_who_knows_everything(toy):
    session = Session(toy)
    while not session.finished:
        session.answer(session.current.answer)
    assert session.verdict.status == SUPPORTED
    assert len(session.history) <= toy.max_questions


def test_session_refutes_a_person_who_knows_nothing(toy):
    session = Session(toy)
    while not session.finished:
        session.answer("B")
    assert session.verdict.status == REFUTED


def test_session_rejects_invalid_answers_and_late_answers(toy):
    session = Session(toy)
    with pytest.raises(ValueError):
        session.answer("Z")
    while not session.finished:
        session.answer("C")
    with pytest.raises(SessionError):
        session.answer("A")


def test_greedy_stops_once_a_claim_is_refuted(toy):
    session = Session(toy, policy=GreedyPolicy())
    while not session.finished:
        session.answer("B")
    claims_asked = {t.probe.claim_id for t in session.history}
    assert claims_asked == {"capitals"}      # one refuted claim settles the verdict


def test_no_claims_is_an_error(toy):
    with pytest.raises(ValueError):
        Session(toy, {"topics": "unknown"})


def test_respondent_answers_by_its_real_level():
    rng = random.Random(0)
    p = Probe(id="p", claim_id="c", question="?", answer="ok", pass_rates=[0.0, 1.0])
    assert StatisticalRespondent({"c": 1}, rng, spread=0).answer(p) == "ok"
    assert StatisticalRespondent({"c": 0}, rng, spread=0).answer(p) != "ok"


def test_true_levels_accepts_bools_and_levels():
    yes_no = Claim("a", "yes/no claim")
    leveled = Claim("b", "skill", levels=("none", "junior", "senior"), claimed_level=2)
    assert true_levels({"a": True, "b": 1}, [yes_no, leveled]) == {"a": 1, "b": 1}
    assert true_levels({"a": False, "b": True}, [yes_no, leveled]) == {"a": 0, "b": 2}


def test_leveled_claim_needs_evidence_at_the_claimed_level():
    claim = Claim("s", "Knows SQL", levels=("none", "junior", "mid", "senior"), claimed_level=3)
    belief = BeliefModel()
    belief.start([claim])
    assert belief.probability("s") == pytest.approx(0.5)      # prior: 50% that it holds
    easy = Probe(id="e", claim_id="s", question="?", pass_rates=irt_pass_rates(4, difficulty=0.5))
    hard = Probe(id="h", claim_id="s", question="?", pass_rates=irt_pass_rates(4, difficulty=2.5))
    assert belief.expected_gain(hard) > belief.expected_gain(easy)
    for _ in range(3):
        belief.update(easy, Observation("e", "s", "x", 1.0))
    assert belief.level("s") >= 1 and belief.status("s") != SUPPORTED   # easy passes are not enough
    for _ in range(3):
        belief.update(hard, Observation("h", "s", "x", 1.0))
    assert belief.status("s") == SUPPORTED and belief.level("s") == 3


def test_irt_pass_rates_rise_with_level():
    rates = irt_pass_rates(5, difficulty=2)
    assert rates == sorted(rates) and rates[2] == pytest.approx(0.5)


def test_greedy_beats_random_on_simulated_people(toy):
    greedy = evaluate(toy, "greedy", episodes=200, seed=3)
    rand = evaluate(toy, "random", episodes=200, seed=3)
    assert greedy["accuracy"] >= rand["accuracy"] - 0.05
    assert greedy["avg_questions"] <= rand["avg_questions"]


def test_score_of_one_half_is_neutral():
    belief = BeliefModel()
    belief.start([Claim("c", "claim")])
    belief.update(probe(), Observation("p", "c", "x", 0.5))
    assert belief.probability("c") == pytest.approx(0.5)
