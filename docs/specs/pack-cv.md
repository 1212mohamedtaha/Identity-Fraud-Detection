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

Claim ids are slugs of the skill name (`machine-learning`). A claim is "Knows <skill>" plus
"(<level>)" when the LLM gives a level.

Pass rates by difficulty:

| Difficulty | `p_true` | `p_false` |
| --- | --- | --- |
| easy | 0.90 | 0.45 |
| medium | 0.80 | 0.30 |
| hard | 0.65 | 0.15 |

`max_questions = 12`, `show_feedback = True`, `accept = 0.85`, `reject = 0.15`.

## Question bank (`data/skills.json`)
`{key: {name, aliases, topics, questions: [{difficulty, question, answer, keywords}]}}`.
Every model answer must contain all of its keywords (a test checks this).
Skills: Python, SQL, JavaScript, React, Docker, Git, machine learning, REST APIs.

## Simulation
`sample_case`: 3 random bank skills; the CV is "Software engineer with experience in …";
each claim is true with probability 0.7.

## Safety
CV, job and answers are passed to prompts inside tags (`<cv>`, `<job>`, `<answer>`), and
every prompt tells the model to treat them as data. Grading prompts judge technical
content only, not grammar or style.

## Ideas for later
Question bank per job family; voice answers; a final coaching report; IRT (item response
theory) instead of fixed pass rates once real answer data exists.
