"""CV pack: check the skills a CV claims with a short adaptive interview.

This is the reference example of extending the core. Every building block has an
LLM version (better, needs a provider) and an offline version (keyword based, free),
so the pack works with or without an LLM.
"""
import json
import logging
import math
import re
from functools import lru_cache
from pathlib import Path

from ...core.belief import irt_pass_rates
from ...core.graph import KnowledgeGraph
from ...core.interfaces import Assessor, ClaimExtractor, KnowledgeSource, ProbeGenerator
from ...core.pack import DomainPack
from ...core.types import Claim, Observation, Probe
from ...llm import LLMError, ask_json

log = logging.getLogger("verity.cv")
HERE = Path(__file__).parent
PROMPTS = HERE / "prompts"
MAX_CLAIMS = 4

# Skill levels, lowest first. A claim "knows SQL at mid level" holds for mid and senior.
LEVELS = ("none", "beginner", "junior", "mid", "senior")
DEFAULT_LEVEL = "junior"          # assumed when the CV does not say
LEVEL_WORDS = {
    "beginner": "beginner", "basic": "beginner", "entry-level": "beginner",
    "junior": "junior",
    "mid": "mid", "mid-level": "mid", "intermediate": "mid",
    "senior": "senior", "expert": "senior", "advanced": "senior", "lead": "senior",
}

# The level at which someone has a 50/50 chance of passing a question of each difficulty.
DIFFICULTY_LEVEL = {"easy": 1.5, "medium": 2.5, "hard": 3.5}


def level_index(word):
    return LEVELS.index(LEVEL_WORDS.get(str(word).lower().strip(), DEFAULT_LEVEL))


def skill_claim(claim_id, skill, level_word, evidence="", bank=None):
    level = level_index(level_word)
    return Claim(id=claim_id, text=f"Knows {skill} at {LEVELS[level]} level", kind="skill",
                 levels=LEVELS, claimed_level=level,
                 data={"skill": skill, "level": LEVELS[level], "evidence": evidence, "bank": bank})


@lru_cache(maxsize=None)
def skill_bank():
    with open(HERE / "data" / "skills.json", encoding="utf-8") as f:
        return json.load(f)


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def bank_key(skill_name):
    """Find the question-bank entry for a skill name, or None."""
    name = skill_name.lower().strip()
    for key, skill in skill_bank().items():
        if name == key or name == skill["name"].lower() or name in skill["aliases"]:
            return key
    return None


def find(text, alias):
    """``(start, end)`` of ``alias`` as a whole word in ``text``, or None."""
    match = re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", text)
    return match.span() if match else None


def mentions(text, alias):
    return find(text, alias) is not None


def level_near(text, span):
    """The level word directly before a skill ("senior Python", "expert in Git") or directly
    after it ("Python (senior)", "Python - senior"). Defaults to DEFAULT_LEVEL."""
    start, end = span
    before = re.search(r"([a-z-]+)\s+(?:(?:in|with|at)\s+)?$", text[:start])
    if before and before.group(1) in LEVEL_WORDS:
        return before.group(1)
    after = re.match(r"\s*[(:–-]?\s*([a-z-]+)", text[end:])
    if after and after.group(1) in LEVEL_WORDS:
        return after.group(1)
    return DEFAULT_LEVEL


# ---------------------------------------------------------------- claims
class CVClaims(ClaimExtractor):
    def __init__(self, llm=None):
        self.llm = llm

    def extract(self, inputs):
        cv = (inputs.get("cv") or "").strip()
        if not cv:
            raise ValueError("Paste a CV to check.")
        job = (inputs.get("job") or "").strip()
        if self.llm:
            try:
                return self.extract_with_llm(cv, job)
            except LLMError as exc:
                log.warning("LLM claim extraction failed, using keywords: %s", exc)
        return self.extract_with_keywords(cv, job)

    def extract_with_llm(self, cv, job):
        reply = ask_json(self.llm, PROMPTS / "extract_claims.v1.md",
                         cv=cv, job=job or "(none)", max_claims=MAX_CLAIMS)
        claims = []
        for item in reply.get("claims", [])[:MAX_CLAIMS]:
            skill = str(item.get("skill", "")).strip()
            if not skill or any(c.id == slug(skill) for c in claims):
                continue
            claims.append(skill_claim(slug(skill), skill, item.get("level", DEFAULT_LEVEL),
                                      str(item.get("evidence", "")), bank_key(skill)))
        return claims

    def extract_with_keywords(self, cv, job):
        cv_text, job_text = cv.lower(), job.lower()
        found = []
        for key, skill in skill_bank().items():
            spans = [find(cv_text, alias) for alias in skill["aliases"]]
            spans = [span for span in spans if span]
            if spans:
                wanted = any(mentions(job_text, alias) for alias in skill["aliases"])
                found.append((not wanted, key, min(spans)))     # skills the job wants come first
        claims = []
        for _, key, span in sorted(found)[:MAX_CLAIMS]:
            name = skill_bank()[key]["name"]
            claims.append(skill_claim(key, name, level_near(cv_text, span), bank=key))
        return claims


# ---------------------------------------------------------------- knowledge
class SkillGraph(KnowledgeSource):
    """Skill -> topic graph from the bundled skill bank."""

    def build(self, claims, inputs):
        graph = KnowledgeGraph(data={"job": inputs.get("job", "")})
        for claim in claims:
            graph.add_node(claim.id, claim.data["skill"], kind="skill")
            key = claim.data.get("bank")
            if key:
                for topic in skill_bank()[key]["topics"]:
                    topic_id = f"{claim.id}/{slug(topic)}"
                    graph.add_node(topic_id, topic, kind="topic")
                    graph.add_edge(claim.id, "has_topic", topic_id, provenance="skills.json")
        return graph


# ---------------------------------------------------------------- questions
def make_probe(probe_id, claim_id, difficulty, question, answer, key_points):
    difficulty = difficulty if difficulty in DIFFICULTY_LEVEL else "medium"
    rates = irt_pass_rates(len(LEVELS), DIFFICULTY_LEVEL[difficulty])
    return Probe(id=probe_id, claim_id=claim_id, question=question, answer=answer,
                 rubric="; ".join(key_points), difficulty=difficulty, pass_rates=rates,
                 data={"key_points": list(key_points)})


class InterviewQuestions(ProbeGenerator):
    def __init__(self, llm=None, per_claim=3):
        self.llm = llm
        self.per_claim = per_claim

    def generate(self, claims, graph):
        if self.llm:
            try:
                probes = self.generate_with_llm(claims, graph)
                if probes:
                    return probes
            except LLMError as exc:
                log.warning("LLM question generation failed, using the question bank: %s", exc)
        return self.generate_from_bank(claims)

    def generate_with_llm(self, claims, graph):
        listed = "\n".join(
            f"- claim_id: {c.id}; skill: {c.data['skill']}; level: {c.data['level']}; "
            f"evidence: {c.data['evidence']}" for c in claims)
        reply = ask_json(self.llm, PROMPTS / "generate_questions.v1.md",
                         claims=listed, job=graph.data.get("job") or "(none)", per_claim=self.per_claim)
        known = {c.id for c in claims}
        probes = []
        for i, item in enumerate(reply.get("questions", [])):
            claim_id = item.get("claim_id")
            if claim_id not in known or not item.get("question"):
                continue
            probes.append(make_probe(f"{claim_id}/llm-{i}", claim_id, item.get("difficulty", "medium"),
                                     item["question"], item.get("answer", ""), item.get("key_points", [])))
        return probes

    def generate_from_bank(self, claims):
        probes = []
        for claim in claims:
            key = claim.data.get("bank")
            if not key:
                continue
            for i, q in enumerate(skill_bank()[key]["questions"]):
                probes.append(make_probe(f"{claim.id}/{i}", claim.id, q["difficulty"],
                                         q["question"], q["answer"], q["keywords"]))
        return probes


# ---------------------------------------------------------------- grading
def keyword_score(answer, key_points):
    """Share of key points mentioned; covering 60% of them already counts as a full pass."""
    text = answer.lower()
    hits = [k for k in key_points if k.lower() in text]
    missing = [k for k in key_points if k not in hits]
    if not key_points:
        return 0.0, "No key points to grade against."
    score = min(1.0, len(hits) / math.ceil(0.6 * len(key_points)))
    feedback = "Covered: " + (", ".join(hits) or "none") + "."
    if missing:
        feedback += " Could also mention: " + ", ".join(missing) + "."
    return score, feedback


class AnswerGrader(Assessor):
    def __init__(self, llm=None):
        self.llm = llm

    def assess(self, probe, answer):
        answer = str(answer).strip()
        key_points = probe.data.get("key_points", [])
        if self.llm:
            try:
                reply = ask_json(self.llm, PROMPTS / "grade_answer.v1.md",
                                 question=probe.question, reference=probe.answer,
                                 key_points="; ".join(key_points), answer=answer)
                score = min(max(float(reply["score"]), 0.0), 1.0)
                return Observation(probe.id, probe.claim_id, answer, score, str(reply.get("feedback", "")))
            except (LLMError, KeyError, TypeError, ValueError) as exc:
                log.warning("LLM grading failed, using keywords: %s", exc)
        score, feedback = keyword_score(answer, key_points)
        return Observation(probe.id, probe.claim_id, answer, score, feedback)


# ---------------------------------------------------------------- the pack
class CVPack(DomainPack):
    name = "cv"
    title = "CV skill check"
    description = ("Paste your CV (and optionally a job description). "
                   "Answer a short interview about the skills you claim.")
    input_fields = [
        {"name": "cv", "label": "Your CV", "type": "textarea", "required": True,
         "placeholder": "Paste the text of your CV…"},
        {"name": "job", "label": "Job description (optional)", "type": "textarea", "required": False,
         "placeholder": "Paste the job you are applying for…"},
    ]
    max_questions = 12
    show_feedback = True
    accept = 0.85
    reject = 0.15

    def claim_extractor(self):
        return CVClaims(self.llm)

    def knowledge_source(self):
        return SkillGraph()

    def probe_generator(self):
        return InterviewQuestions(self.llm)

    def assessor(self):
        return AnswerGrader(self.llm)

    def sample_case(self, rng):
        """A simulated candidate: real levels per skill; 60% describe themselves honestly,
        the rest claim one or two levels more than they have."""
        keys = rng.sample(sorted(skill_bank()), 3)
        honest = rng.random() < 0.6
        parts, truth = [], {}
        for key in keys:
            real = rng.randint(1, len(LEVELS) - 1)
            claimed = real if honest else min(real + rng.randint(1, 2), len(LEVELS) - 1)
            truth[key] = real
            parts.append(f"{LEVELS[claimed]} {skill_bank()[key]['name']}")
        return {"cv": "Software engineer. Skills: " + ", ".join(parts) + "."}, truth
