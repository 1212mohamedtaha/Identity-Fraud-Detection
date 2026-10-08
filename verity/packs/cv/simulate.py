"""Synthetic candidates for the CV pack: personas, their answers, and a mock LLM.

Everything here is made up on purpose, to train and test policies without real people or
API calls. The synthetic world deliberately differs from what the belief model assumes
(hidden question difficulty, sharper or weaker people, tiredness, looking answers up,
paraphrased answers), otherwise there would be nothing for a policy to learn.
See docs/specs/dataset.md.
"""
import hashlib
import json
import math
import re

from ...core.dataset import write_jsonl
from ...llm.base import LLM
from . import DIFFICULTY_LEVEL, LEVELS, CVClaims, keyword_score, skill_bank

# Share of candidates per honesty type.
HONESTY = {"genuine": 0.6, "exaggerator": 0.3, "impostor": 0.1}
SKILLS_PER_CV = 3
UNSURE_LINES = ["I'm not sure.", "I don't know, sorry.", "No idea, I haven't used that much."]
PARAPHRASE = "something related to that"   # stands in for a key point said in other words


def stable_unit(text):
    """A number in [0, 1) that depends only on ``text`` (stable across runs)."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) / 16 ** 8


def hidden_difficulty(question, label):
    """The question's real difficulty: its label's level plus a fixed offset in [-0.75, 0.75].
    The belief model only knows the label, so this is one way the world differs from it."""
    return DIFFICULTY_LEVEL.get(label, 2.5) + (stable_unit(question) - 0.5) * 1.5


# ---------------------------------------------------------------- personas
def make_persona(rng, index):
    honesty = rng.choices(list(HONESTY), weights=list(HONESTY.values()))[0]
    keys = rng.sample(sorted(skill_bank()), SKILLS_PER_CV)
    top = len(LEVELS) - 1
    real, claimed, parts = {}, {}, []
    for key in keys:
        if honesty == "impostor":
            real[key] = rng.randint(0, 1)
            claimed[key] = rng.randint(3, top)
        else:
            real[key] = rng.randint(1, top)
            bump = rng.randint(1, 2) if honesty == "exaggerator" else 0
            claimed[key] = min(real[key] + bump, top)
        name, level = skill_bank()[key]["name"], LEVELS[claimed[key]]
        parts.append(rng.choice([f"{level} {name}", f"{name} ({level})"]))
    cv = rng.choice(["Software engineer. Skills: ", "Developer with experience in: ", "Profile - "])
    return {
        "id": f"cv-{index:05d}",
        "inputs": {"cv": cv + ", ".join(parts) + "."},
        "truth": real,
        "claimed": claimed,
        "honesty": honesty,
        "traits": {
            "sharpness": round(rng.gauss(0, 0.35), 3),       # added to every skill level
            "fatigue": round(rng.uniform(0, 0.06), 3),        # level lost per question asked
            "lookup": 0.35 if honesty == "impostor" else 0.03,  # chance to look an answer up
        },
    }


def answer_quality(persona, skill, question, label, asked, rng):
    """How good (0..1) this persona's answer to ``question`` is, after ``asked`` questions."""
    traits = persona["traits"]
    ability = persona["truth"].get(skill, 0) + traits["sharpness"] - traits["fatigue"] * asked
    expected = 0.05 + 0.9 / (1 + math.exp(-1.7 * (ability - hidden_difficulty(question, label))))
    if rng.random() < traits["lookup"]:
        expected = max(expected, rng.uniform(0.55, 0.85))
    return min(max(expected + rng.gauss(0, 0.12), 0.0), 1.0)


def write_answer(key_points, quality, rng):
    """Answer text covering about ``quality`` of the key points; some are paraphrased, so a
    keyword grader will miss them while a good (mock) LLM grader still credits them."""
    if quality < 0.15 or not key_points:
        return rng.choice(UNSURE_LINES)
    count = max(1, round(quality * len(key_points)))
    said = [PARAPHRASE if rng.random() < 0.2 else point for point in rng.sample(key_points, count)]
    text = "I'd say it comes down to " + ", ".join(said) + "."
    if quality < 0.5:
        text += " But I'm honestly not sure about the details."
    return text


class PersonaRespondent:
    """Plays a persona in a session: writes answers and remembers their true quality, so the
    mock LLM grader can grade them like a good but imperfect human would."""

    def __init__(self, persona, rng, quality_index=None):
        self.persona = persona
        self.rng = rng
        self.asked = 0
        self.quality_index = quality_index if quality_index is not None else {}

    def answer(self, probe):
        quality = answer_quality(self.persona, probe.claim_id, probe.question, probe.difficulty,
                                 self.asked, self.rng)
        self.asked += 1
        text = write_answer(probe.data.get("key_points", []), quality, self.rng)
        self.quality_index[text] = quality
        return text


# ---------------------------------------------------------------- mock LLM
def between(text, start, end):
    match = re.search(re.escape(start) + r"(.*?)" + re.escape(end), text, re.S)
    return match.group(1).strip() if match else ""


class MockInterviewLLM(LLM):
    """Answers the CV pack's three prompts without any API, so simulations exercise the
    full LLM path (prompt files, JSON parsing, fallbacks):

    - extract_claims: reads the CV with the offline extractor;
    - generate_questions: returns the question bank for the listed claims;
    - grade_answer: the answer's true quality (if this mock saw it being written) plus
      grading noise, else key-point coverage. Like an LLM judge: good, not perfect.
    """

    provider = "mock"
    model = "mock-interviewer"

    def __init__(self, rng, noise=0.08):
        self.rng = rng
        self.noise = noise
        self.quality_index = {}       # answer text -> true quality, filled by PersonaRespondent

    def complete(self, system, prompt, max_tokens=4000):
        if "extract skill claims" in system:
            cv, job = between(prompt, "<cv>", "</cv>"), between(prompt, "<job>", "</job>")
            claims = CVClaims().extract_with_keywords(cv, job)
            return json.dumps({"claims": [{"skill": c.data["skill"], "level": c.data["level"], "evidence": ""}
                                          for c in claims]})
        if "interviewer" in system:
            questions = []
            for claim_id in re.findall(r"claim_id: ([a-z0-9-]+)", prompt):
                for q in skill_bank().get(claim_id, {}).get("questions", []):
                    questions.append({"claim_id": claim_id, "difficulty": q["difficulty"], "question": q["question"],
                                      "answer": q["answer"], "key_points": q["keywords"]})
            return json.dumps({"questions": questions})
        if "grade interview answers" in system:
            answer = between(prompt, "<answer>", "</answer>")
            quality = self.quality_index.get(answer)
            if quality is None:
                points = [p.strip() for p in between(prompt, "Key points:", "<answer>").split(";") if p.strip()]
                quality = keyword_score(answer, points)[0]
            score = min(max(quality + self.rng.gauss(0, self.noise), 0.0), 1.0)
            return json.dumps({"score": round(score, 3), "feedback": "Mock grade."})
        return "{}"


# ---------------------------------------------------------------- dataset files
def write_answers(cases, out, rng):
    """answers.jsonl: every case answering every bank question of its skills, each at a random
    position in a session (0 = first question, so fatigue applies), with the true quality.
    Used to fit question parameters, fatigue and person spread, and to measure graders.
    questions.json: each bank question's hidden difficulty."""
    rows, questions = [], {}
    for case in cases:
        for skill in case["truth"]:
            for i, q in enumerate(skill_bank()[skill]["questions"]):
                position = rng.randrange(12)
                quality = answer_quality(case, skill, q["question"], q["difficulty"], position, rng)
                rows.append({
                    "case": case["id"], "split": case["split"], "skill": skill, "question_id": f"{skill}/{i}",
                    "level": case["truth"][skill], "position": position, "difficulty": q["difficulty"],
                    "quality": round(quality, 3), "answer": write_answer(q["keywords"], quality, rng),
                })
                hidden = hidden_difficulty(q["question"], q["difficulty"])
                questions[f"{skill}/{i}"] = {"question": q["question"], "label": q["difficulty"],
                                             "hidden_difficulty": round(hidden, 3)}
    write_jsonl(out / "answers.jsonl", rows)
    (out / "questions.json").write_text(json.dumps(questions, indent=2, ensure_ascii=False))
