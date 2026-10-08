import random

import pytest

from verity.core.belief import BeliefModel, irt_pass_rates
from verity.core.engine import Session, SessionError
from verity.core.policies import GreedyPolicy
from verity.core.simulation import StatisticalRespondent, evaluate, true_levels
from verity.core.types import REFUTED, SUPPORTED, Choice, Claim, Observation, Probe


def probe(claim_id="c", p_true=0.9, p_false=0.3, free_text=False):
    choices = [] if free_text else [Choice("A", "yes"), Choice("B", "no")]
    return Probe(id="p", claim_id=claim_id, question="?", choices=choices, p_true=p_true, p_false=p_false)


def test_belief_moves_up_on_pass_and_down_on_fail():
    belief = BeliefModel()
    belief.start([Claim("c", "claim")])
    belief.update(probe(), Observation("p", "c", "x", 1.0))
    assert belief.probability("c") > 0.5
    belief.start([Claim("c", "claim")])
    belief.update(probe(), Observation("p", "c", "x", 0.0))
    assert belief.probability("c") < 0.5


def test_free_text_score_counts_by_distance_to_expected_scores():
    """Expected scores 0.3 (claim false) and 0.9 (true): 0.8 supports, 0.6 is neutral, 0.4 refutes."""
    results = {}
    for score in (0.8, 0.6, 0.4):
        b = BeliefModel()
        b.start([Claim("c", "claim")])
        b.update(probe(free_text=True), Observation("p", "c", "x", score))
        results[score] = b.probability("c")
    assert results[0.8] > 0.5 and results[0.4] < 0.5
    assert results[0.6] == pytest.approx(0.5)


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


def test_partial_score_favours_the_level_that_expects_it():
    """Regression: a 0.32 on a question where mid-level people average 0.32 must support
    "mid", not count as a fail that favours "junior" (found by the calibration check)."""
    claim = Claim("s", "skill", levels=("junior", "mid", "senior"), claimed_level=1)
    belief = BeliefModel()
    belief.start([claim])
    hard = Probe(id="h", claim_id="s", question="?", pass_rates=[0.12, 0.32, 0.68])     # free text
    for _ in range(4):
        belief.update(hard, Observation("h", "s", "x", 0.32))
    assert belief.level("s") == 1 and belief.probability("s") > 0.5


def test_person_factor_lets_one_claim_inform_another():
    """Someone who aces one skill is probably sharp: their other skill's expected level rises."""
    levels = ("none", "junior", "mid", "senior")
    claims = [Claim("a", "A", levels=levels, claimed_level=2), Claim("b", "B", levels=levels, claimed_level=2)]
    hard = Probe(id="h", claim_id="a", question="?", pass_rates=irt_pass_rates(4, 2.5))
    plain, with_person = BeliefModel(), BeliefModel(person_spread=0.5)
    for belief in (plain, with_person):
        belief.start(claims)
        for _ in range(3):
            belief.update(hard, Observation("h", "a", "x", 0.95))
    assert plain.probability("b") == pytest.approx(0.5)
    assert with_person.person_offset() > 0
    assert BeliefModel(person_spread=0).person_offset() == 0


def test_no_person_factor_and_no_fatigue_is_the_plain_model():
    claim = Claim("c", "claim")
    a, b = BeliefModel(), BeliefModel(person_spread=0.0, fatigue=0.0)
    for belief in (a, b):
        belief.start([claim])
        belief.update(probe(), Observation("p", "c", "x", 1.0))
    assert a.probability("c") == b.probability("c")


def test_fatigue_makes_late_failures_count_less():
    claim = Claim("s", "skill", levels=("none", "junior", "mid"), claimed_level=2)
    q = Probe(id="q", claim_id="s", question="?", pass_rates=irt_pass_rates(3, 1.5))
    rested, tired = BeliefModel(), BeliefModel(fatigue=0.1)
    for belief in (rested, tired):
        belief.start([claim])
        for _ in range(5):
            belief.update(q, Observation("q", "s", "x", 0.4))
    assert tired.probability("s") > rested.probability("s")


def test_gap_prior_sets_the_starting_belief():
    claim = Claim("s", "skill", levels=("none", "junior", "mid", "senior"), claimed_level=2)
    belief = BeliefModel(gap_prior={0: 0.7, -1: 0.2, -2: 0.1})
    belief.start([claim])
    assert belief.probability("s") == pytest.approx((0.7 + 0.01 + 0.01) / (0.7 + 0.2 + 0.1 + 0.04))
    assert belief.level("s") == 2
