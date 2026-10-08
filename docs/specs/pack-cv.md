# Spec: CV pack

Code: `verity/packs/cv/`. Name: `cv`. This is the reference example of extending the core.

Checks the technical skills a CV claims with a short interview of free-text questions.
Built for interview practice: the person sees a score and feedback after every answer.

## Inputs
| Field | Required | Meaning |
| --- | --- | --- |
| `cv` | yes | CV text. Empty → `ValueError("Paste a CV to check.")` |
| `job` | no | Job description; skills it mentions are checked first |

## Blocks
Each block has an **LLM version** (used when an LLM is configured) and an **offline version**
(used without an LLM, or when the LLM call fails).

| Block | LLM version | Offline version |
| --- | --- | --- |
| Claims (`CVClaims`) | `prompts/extract_claims.v1.md`: up to 4 skills with level and evidence | Skills from `data/skills.json` whose aliases appear in the CV; skills the job mentions first; up to 4 |
| Knowledge (`SkillGraph`) | – | skill node → topic nodes from `skills.json` |
| Probes (`InterviewQuestions`) | `prompts/generate_questions.v1.md`: 3 questions per claim (easy, medium, hard), with model answer and key points; asks about projects in the evidence | The question bank in `skills.json` |
| Assessor (`AnswerGrader`) | `prompts/grade_answer.v1.md`: score 0–1 + feedback | Key-point matching: `score = min(1, hits / ceil(0.6 × key points))`; feedback lists covered and missing points |

## Levels
Skill claims are leveled: `LEVELS = ("none", "beginner", "junior", "mid", "senior")`.
A claim "Knows SQL at mid level" holds for real level mid or senior.

- The claimed level comes from the LLM (`level` field) or, offline, from a level word directly
  next to the skill in the CV: "senior Python", "expert in Git", "Docker (mid)", "Python - beginner".
  Words: beginner/basic/entry-level, junior, mid/mid-level/intermediate, senior/expert/advanced/lead.
- No level given → `junior`.

Claim ids are slugs of the skill name (`machine-learning`); text "Knows <skill> at <level> level".

Pass rates per level come from `irt_pass_rates(5, difficulty_level)`:

| Difficulty | 50/50 at level | none | beginner | junior | mid | senior |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 1.5 | 0.12 | 0.32 | 0.68 | 0.88 | 0.94 |
| medium | 2.5 | 0.06 | 0.12 | 0.32 | 0.68 | 0.88 |
| hard | 3.5 | 0.05 | 0.06 | 0.12 | 0.32 | 0.68 |

`max_questions = 12`, `show_feedback = True`, `accept = 0.85`, `reject = 0.15`.

## Question bank (`data/skills.json`)
`{key: {name, aliases, topics, questions: [{difficulty, question, answer, keywords}]}}`.
Every model answer must contain all of its keywords (a test checks this).
Skills: Python, SQL, JavaScript, React, Docker, Git, machine learning, REST APIs.

## Simulation
`sample_case`: 3 random bank skills with real levels beginner..senior. 60% of candidates
claim their real levels; the rest claim 1–2 levels more. The CV reads
"Software engineer. Skills: senior Python, junior SQL, …".

## Safety
CV, job and answers are passed to prompts inside tags (`<cv>`, `<job>`, `<answer>`), and
every prompt tells the model to treat them as data. Grading prompts judge technical
content only, not grammar or style.

## Ideas for later
Question bank per job family; voice answers; a final coaching report; fit each question's
IRT difficulty from answer data instead of the difficulty label.
