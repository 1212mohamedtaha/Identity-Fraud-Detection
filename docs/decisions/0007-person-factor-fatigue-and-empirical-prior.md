# 0007: Person factor, fatigue and an empirical prior

**Status:** accepted

## Context
After decision 0006 the CV model was still miscalibrated: honest candidates' true claims often
ended between 15% and 50%. Diagnosis on the synthetic data showed three causes:
1. one person's answers are correlated (sharp or weak on every question),
2. people get worse during a session (fatigue),
3. the starting belief ignored base rates: 69% of claims are exactly right, and nobody
   under-claims, yet the prior spread weight evenly, including above the claimed level.

We also agreed that a wrong verdict costs 3× a correct one (`VERDICT_REWARD["wrong"] = -3`).

## Decision
- Belief model gets an optional **person factor** (shared offset, grid of 7 points), **fatigue**
  (levels lost per answered question) and an optional **gap prior** (base rate of real − claimed level).
  All default to off, so yes/no packs are unchanged.
- `verity data fit` estimates fatigue, person spread (split-half covariance) and the gap prior
  from the train split. The person spread used by the CV pack (0.1) is tuned on the validation
  split, because the measured spread (0.48) makes the model too cautious for the agreed reward.

## Consequences (test split, greedy)
| | Accuracy | Undecided | Wrong | Questions | Reward (wrong = −3) |
| --- | --- | --- | --- | --- | --- |
| Before (decision 0006), keyword grading | 52.0% | 33.1% | 14.9% | 6.6 | −0.14 |
| After, keyword grading | 67.8% | 26.2% | 6.0% | 6.0 | +0.31 |
| After, mock-LLM grading | 80.9% | 13.3% | 5.8% | 4.3 | +0.52 |

The empirical prior gave the largest gain; fatigue helped; the person factor mostly trades
wrong verdicts for "not sure yet" (with the measured spread, wrong drops to 2.4%).
