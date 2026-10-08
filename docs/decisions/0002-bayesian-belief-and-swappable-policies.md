# 0002: Bayesian belief per claim; policies only choose questions

**Status:** accepted

## Context
The original model decided "fraud / not fraud" directly from its policy network. That gave
no probability, no explanation, and could not be checked without real labelled data.

## Decision
- A **belief model** keeps a probability per claim, updated with Bayes' rule from each
  graded answer, using the probe's `p_true` / `p_false`. It alone produces the verdict.
- A **policy** only decides which question to ask next or when to stop. Policies are
  swappable: `greedy` (information-based), `random`, `learned` (RL), `legacy` (original model).
- Every policy is measured the same way on simulated people (`verity evaluate`), with one
  reward that RL training also maximises.

## Consequences
- Verdicts come with probabilities and plain explanations.
- RL has a clear, measurable job and a strong baseline to beat.
- Results are only as realistic as `p_true` / `p_false`; real data should replace guesses.
