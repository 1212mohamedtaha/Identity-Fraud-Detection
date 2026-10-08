# Spec: reinforcement learning

Code: `verity/rl/`. Guide: [../guides/training.md](../guides/training.md).

## What is learned

A **policy**: at each step, which open probe to ask next, or stop. The belief model and
the verdict rules stay fixed; RL only learns the questioning strategy.

## Actions

Candidates are `open_candidates(state)` (unasked probes of undecided claims) plus **stop**.
If there are no candidates, or the verdict is already settled (a claim is refuted), the
policy stops without consulting the network.

## Features (`features.py`)

Per candidate probe (8 numbers): claim probability; how undecided the claim is
(`1 - |p - 0.5| * 2`); expected information gain; `p_true`; `p_false`; questions already
asked on this claim / `max_questions`; questions asked in total / `max_questions`;
1 if the claim is undecided.

For stop (4 numbers): share of decided claims; questions asked / `max_questions`;
lowest claim probability; highest expected gain among unasked probes.

The features use only beliefs and probe pass rates, so the same network works for any pack.

## Network (`policy.py`)

`PolicyNet`: two small MLPs (`Linear(F, 32) → Tanh → Linear(32, 1)`), one scoring each
candidate and one scoring stop. A softmax over `[candidate scores…, stop score]` gives the
action probabilities. `LearnedPolicy(net, explore=True)` samples and records
log-probabilities and entropies; with `explore=False` it takes the arg-max.

Saved file: `torch.save({"state_dict": …, "info": {"pack", "episodes", "question_cost"}})`,
loaded with `weights_only=True`. Default path: `models/<pack>/policy.pt` (relative to the
working directory; `models/` is git-ignored).

## Training (`train.py`)

REINFORCE with a running-average baseline and an entropy bonus:

```
for each episode:
    sample a simulated person (pack.sample_case) and play a full session with explore=True
    reward = episode_reward(session, truth)          # +1 / -0.25 / -1, minus 0.02 per question
    loss   = -(reward - baseline) * sum(log_probs) - entropy_bonus * sum(entropies)
    Adam step; baseline = 0.95 * baseline + 0.05 * reward
```

Defaults: `episodes=2000`, `lr=0.01`, `entropy_bonus=0.01`, `question_cost=0.02`, `seed=0`.

## Evaluation

`verity evaluate <pack>` runs every available policy on the same simulated people
(same seed) and reports accuracy, uncertain, wrong, average questions and average reward.
A new policy is only an improvement if its **reward** is higher.

## Current results and limits

With default settings (`verity train <pack>`, 2000 episodes, then `verity evaluate <pack>`):

| Pack | greedy reward | learned reward | Notes |
| --- | --- | --- | --- |
| identity | +0.262 | **+0.311** | learned is more accurate (65.7% vs 61.7%) and wrong less often (6.3% vs 10.0%), at ~1.4 more questions |
| cv | **+0.319** | +0.298 | close; greedy asks fewer questions |

Results move by a few points with the seed and episode count. The gains are modest
because the simulator uses the same pass rates as the belief model, and under those
assumptions the greedy rule is already close to optimal. RL should pay off more once the
simulator reflects real behaviour that the simple rule ignores (people who tire, topics
that hang together, questions that are easy to look up). Next steps:

1. Log real sessions (with consent) and fit per-probe pass rates from them.
2. Make the respondent model richer (correlated knowledge, fatigue, look-up behaviour).
3. Then retrain and compare with `verity evaluate`.
