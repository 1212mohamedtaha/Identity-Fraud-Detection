# Spec: synthetic datasets (datasheet)

Code: `verity/core/dataset.py` (generic), `verity/packs/cv/simulate.py` (CV personas).
Command: `verity data generate <pack> --size 3000 --seed 0 [--out datasets/<pack>]`.

> **Everything here is synthetic.** No real person is in any dataset. Results measured on
> it say whether one method beats another *under these assumptions*, not how accurate
> Verity is on real people.

## Why it exists
There is no labelled real data yet. To train and compare policies, fit model parameters and
test graders we need people whose truth is known. The generated world is built to **differ
from what the belief model assumes**, otherwise the simple greedy rule would already be
optimal and there would be nothing to learn or measure.

## Files

| File | Content |
| --- | --- |
| `cases.jsonl` | One simulated person per line (all packs). |
| `info.json` | `{pack, size, seed, splits}`. |
| `answers.jsonl` | CV only: every person's answer to every bank question of their skills. |
| `questions.json` | CV only: each bank question's label and hidden true difficulty. |

Splits are deterministic by position: first 70% `train`, next 15% `val`, last 15% `test`.
The same `--size` and `--seed` always produce the same files.
`datasets/` is git-ignored, except the small committed example `datasets/cv-sample/` (60 people).

## Case (all packs)
`{"id", "split", "inputs", "truth"}`. `inputs` start a session; `truth` maps claim id → real
level (or bool for yes/no claims). Packs create cases with `DomainPack.make_case(rng, index)`;
the default wraps `sample_case`.

## CV personas (`make_persona`)

| Field | How it is generated |
| --- | --- |
| `honesty` | `genuine` 60%, `exaggerator` 30%, `impostor` 10% |
| skills | 3 distinct skills from the question bank |
| `truth` (real level) | genuine / exaggerator: beginner..senior uniformly; impostor: none or beginner |
| `claimed` | genuine: = real; exaggerator: real + 1..2 (max senior); impostor: mid or senior |
| `inputs.cv` | e.g. "Software engineer. Skills: senior Python, SQL (junior), mid React." |
| `traits.sharpness` | Normal(0, 0.35), added to every skill level |
| `traits.fatigue` | Uniform(0, 0.06) levels lost per question already asked |
| `traits.lookup` | chance to look an answer up: 0.35 impostors, 0.03 others |

## How a persona answers (`answer_quality`, `write_answer`)

1. Ability = real level + sharpness − fatigue × questions already asked.
2. Each bank question has a **hidden difficulty** = its label's level (easy 1.5, medium 2.5,
   hard 3.5) + a fixed offset in [−0.75, 0.75] derived from the question text.
   The belief model only sees the label.
3. Expected quality = `0.05 + 0.9 × sigmoid(1.7 × (ability − hidden difficulty))`; with
   probability `lookup` it is raised to Uniform(0.55, 0.85). Quality = expected + Normal(0, 0.12),
   clipped to [0, 1].
4. Text: below 0.15, an "I'm not sure" line. Otherwise it mentions about `quality × N` of the
   question's N key points; each is paraphrased with probability 0.2 (written as
   "something related to that"), which keyword grading cannot credit. Below 0.5 it adds
   "But I'm honestly not sure about the details."

So a "pass" is fuzzy, graders disagree, people tire, and impostors sometimes do well.

## Mock LLM (`MockInterviewLLM`)
Plays the CV pack's three prompts so simulations run the full LLM path without an API:
- `extract_claims`: the offline keyword extractor's result, as the LLM's JSON.
- `generate_questions`: the bank questions of the listed claims.
- `grade_answer`: the answer's true quality (known because `PersonaRespondent` records it
  while writing) + Normal(0, 0.08) noise; key-point coverage for answers it did not see.

Use it with `--mock-llm` on `verity evaluate` and `verity train`.

## `answers.jsonl` rows (CV)
`{case, split, skill, question_id, level, difficulty, quality, answer}`: each persona answering
each bank question of its skills as a first question (no fatigue). Used to fit question
parameters and to measure graders.

## Using the dataset
| Command | What it does |
| --- | --- |
| `verity evaluate cv --data datasets/cv [--mock-llm]` | Scores policies on the test split (`--split val` for tuning). |
| `verity train cv --data datasets/cv [--mock-llm]` | Trains the RL policy on the train split. |
| `verity data fit cv --data datasets/cv` | Fits each question's difficulty / discrimination and the score noise on the train split. |
| `verity data grader-eval cv --data datasets/cv` | Compares keyword and mock-LLM grading with the true quality. |

Rule: tune on `val`, report on `test`, never fit or train on `test`.

## Known limits
- Answer texts are templated, not natural language; a real LLM would grade them differently.
- Personas are independent across skills (no "knows Python, so probably knows Django").
- The question bank is small (8 skills × 7 questions).
- All rates and shares above are assumptions, chosen to be plausible, not measured.
- Honest candidates sit exactly at their claimed level, so with person-to-person variation
  many honest claims are genuinely ambiguous; this caps how well any method can do.
