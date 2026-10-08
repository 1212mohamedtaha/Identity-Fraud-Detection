import json

import pytest

from verity.core.engine import Session
from verity.core.types import SUPPORTED
from verity.llm.fake import FakeLLM
from verity.packs.cv import CVPack, keyword_score, skill_bank

CV = "Backend developer: 5 years of Django and PostgreSQL, Docker in production, some React."


def test_every_bank_answer_passes_its_own_keywords():
    for skill in skill_bank().values():
        for q in skill["questions"]:
            assert keyword_score(q["answer"], q["keywords"])[0] == 1.0, q["question"]


def test_offline_claims_put_job_skills_first():
    session = Session(CVPack(), {"cv": CV, "job": "We need strong SQL"})
    ids = [c.id for c in session.state.claims]
    assert ids[0] == "sql"
    assert set(ids) == {"python", "sql", "docker", "react"}


def test_empty_cv_is_rejected():
    with pytest.raises(ValueError):
        Session(CVPack(), {"cv": "  "})


def test_offline_session_with_model_answers_is_supported():
    session = Session(CVPack(), {"cv": "I write Python and SQL."})
    while not session.finished:
        session.answer(session.current.answer)
    assert session.verdict.status == SUPPORTED
    assert all(t.observation.feedback for t in session.history)


def test_llm_versions_of_every_block():
    def reply(system, prompt):
        if "extract skill claims" in system:
            return json.dumps({"claims": [{"skill": "Rust", "level": "senior", "evidence": "wrote a DB"}]})
        if "interviewer" in system:
            return json.dumps({"questions": [
                {"claim_id": "rust", "difficulty": "hard", "question": "Explain ownership.",
                 "answer": "Each value has one owner.", "key_points": ["owner", "borrow"]}]})
        return json.dumps({"score": 0.9, "feedback": "Good."})

    llm = FakeLLM(reply)
    session = Session(CVPack(llm=llm), {"cv": "Senior Rust engineer"})
    assert session.state.claims[0].text == "Knows Rust at senior level"
    assert session.state.claims[0].claimed_level == 4
    assert session.current.question == "Explain ownership."
    observation = session.answer("Each value has an owner; you can borrow it.")
    assert observation.score == 0.9 and observation.feedback == "Good."
    assert "<cv>" in llm.calls[0][1]          # the CV is passed as data inside tags


def test_llm_failures_fall_back_to_offline_logic():
    llm = FakeLLM(lambda system, prompt: "not json at all")
    session = Session(CVPack(llm=llm), {"cv": CV})
    assert session.state.claims and session.current is not None
    observation = session.answer("mutable immutable hashable")
    assert 0.0 <= observation.score <= 1.0


@pytest.mark.parametrize("cv,expected", [
    ("Skills: senior Python, junior SQL, Docker (mid), React",
     {"python": "senior", "sql": "junior", "docker": "mid", "react": "junior"}),
    ("Expert in Git; Python - beginner", {"git": "senior", "python": "beginner"}),
])
def test_offline_extraction_reads_levels_next_to_skills(cv, expected):
    claims = Session(CVPack(), {"cv": cv}).state.claims
    assert {c.id: c.data["level"] for c in claims} == expected


def test_claimed_level_decides_the_verdict():
    """The same answers support a junior claim but not a senior one."""
    def run(level):
        session = Session(CVPack(), {"cv": f"{level} SQL"})
        while not session.finished:
            probe = session.current
            session.answer(probe.answer if probe.difficulty != "hard" else "no idea")
        return session.verdict.claims[0]

    junior, senior = run("junior"), run("senior")
    assert junior.status == "supported"
    assert senior.status != "supported"
    assert junior.level >= junior.claim.claimed_level


@pytest.mark.parametrize("cv,expected", [
    ("SKILLS\nSQL: 5 years, Python (2 years), 4+ years of Docker",
     {"sql": "senior", "python": "junior", "docker": "mid"}),
    ("Skills: REST APIs: 3 years", {"rest-apis": "mid"}),
    ("- Built services in Django\nSKILLS\nPython - advanced", {"python": "senior"}),   # strongest mention wins
])
def test_offline_extraction_reads_years_and_all_mentions(cv, expected):
    claims = Session(CVPack(), {"cv": cv}).state.claims
    assert {c.id: c.data["level"] for c in claims} == expected


def test_generated_cvs_look_real_and_are_read_correctly():
    import random

    from verity.packs.cv.cvgen import FILLER_TOOLS
    from verity.packs.cv.simulate import make_persona

    rng = random.Random(0)
    for i in range(60):
        persona = make_persona(rng, i)
        cv = persona["inputs"]["cv"]
        assert all(section in cv for section in ("SUMMARY", "EXPERIENCE", "EDUCATION", "SKILLS"))
        assert "@example.com" in cv
        claims = Session(CVPack(), persona["inputs"]).state.claims
        assert {c.id: c.claimed_level for c in claims} == persona["claimed"]
    for tool in FILLER_TOOLS:                       # filler tools are never mistaken for checked skills
        with pytest.raises(ValueError):
            Session(CVPack(), {"cv": f"Skills: {tool}"})
