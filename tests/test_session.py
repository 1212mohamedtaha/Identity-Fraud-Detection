"""End-to-end behaviour of a dialogue with the pretrained policy."""
import pytest

from fraud_detection.dialogue import Outcome


def correct_choice(session):
    answer = session.graph.idx2node[str(session._pending.a_node)]
    return next(c["id"] for c in session.current.choices if c["text"] == answer)


def wrong_choice(session):
    answer = session.graph.idx2node[str(session._pending.a_node)]
    return next(c["id"] for c in session.current.choices if c["text"] not in (answer, "Not Sure"))


def play(session, pick):
    for _ in range(100):
        if session.finished:
            return session.current
        session.answer(pick(session))
    pytest.fail("dialogue never ended")


def test_truthful_applicant_is_verified(engine):
    session = engine.new_session()
    turn = play(session, correct_choice)
    assert turn.status == "non_fraud"
    assert turn.matched == turn.answered > 0


@pytest.mark.parametrize("pick", [wrong_choice, lambda s: "D"], ids=["wrong", "not-sure"])
def test_applicant_who_cannot_answer_is_flagged(engine, pick):
    session = engine.new_session()
    turn = play(session, pick)
    assert turn.status == "fraud"
    assert turn.matched == 0


def test_history_records_every_answer(engine):
    session = engine.new_session()
    first_question = session.current.question
    session.answer("a")
    assert session.history[0]["question"] == first_question
    assert session.history[0]["selected"] == "A"


def test_invalid_choice_is_rejected_without_changing_state(engine):
    session = engine.new_session()
    before = session.current
    with pytest.raises(ValueError):
        session.answer("Z")
    assert session.current is before and not session.history


def test_cannot_answer_after_the_dialogue_finished(engine):
    session = engine.new_session()
    play(session, lambda s: "D")
    assert session.outcome is not Outcome.CONTINUE
    with pytest.raises(RuntimeError):
        session.answer("A")


def test_sessions_do_not_share_state(engine):
    first, second = engine.new_session(), engine.new_session()
    first.answer("D")
    assert second.tracker.state.total_qa_turn == 0
    assert not second.history
