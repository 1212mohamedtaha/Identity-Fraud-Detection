# 0006: How a graded answer counts as evidence

**Status:** accepted

## Context
Free-text answers get a score between 0 and 1. We first read a score as "the chance the
answer passed" and mixed the pass and fail cases (`s·r + (1−s)·(1−r)`). Measuring
calibration on the synthetic dataset showed this is **biased**: a mid-level person scoring
the 0.3 typical for mid-level people on a hard question was counted as evidence for
"junior". True claims ended up with low probabilities.

## Decision
- **Multiple choice:** pass/fail. Likelihood at each level is `rate` (pass) or `1 − rate` (fail).
- **Free text:** the score is "expected score at the person's level, plus noise". Likelihood is
  a bell curve around the expected score (`pass_rates[level]`) with width `score_noise`.
- `score_noise` is **measured** from data (`verity data fit cv`): the typical distance between
  observed scores and the fitted expected score: 0.18 for LLM-style grading, 0.27 for keyword
  grading in the current synthetic data.

## Consequences
- The identity pack (multiple choice) is mathematically unchanged; its results are identical.
- CV verdicts come with fewer questions and more decisions (test split, greedy, keyword grading:
  reward +0.04 → +0.16).
- Remaining limit: one person's answers are not independent (a sharp or tired person is so on
  every question), which the model does not represent; it is somewhat overconfident after many
  answers. A person-level factor in the belief model is the principled next step.
- Wrong verdicts rose (10.7% → 14.9%) as the model decides more often. How wrong verdicts should
  be weighed against "not sure yet" is a product decision (the reward weights).
