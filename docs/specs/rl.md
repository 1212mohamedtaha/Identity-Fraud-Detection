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

Domain-agnostic; the full list with formulas is in [../modeling.md](../modeling.md) §9.3.
Per candidate probe (12): claim probability, how undecided it is, expected gain, the probe's
separation around the claimed level, expected score at the claimed level, claimed level,
questions on the claim, mean score on the claim, session progress, undecided flag, free-text
flag, expected person offset. Session (7): share decided, progress, min and mean claim
probability, mean undecidedness, largest gain, expected person offset.

## Network (`policy.py`)

`PolicyNet`: three small MLPs (`Linear(F, 32) → Tanh → Linear(32, 1)`): one scores each
candidate, one scores stop, one (the critic) predicts the return. A softmax over
`[candidate scores…, stop score]` gives the action probabilities.

`LearnedPolicy(net)` takes the best-scored action; `explore=True` samples and records
log-probability, entropy, value and position for training; `imitate=policy` follows another
policy and records the network's log-probability of its choices.

Saved file: `torch.save({"state_dict", "info": {pack, episodes, seed, validation_reward}})`,
loaded with `weights_only=True`. Default path: `models/<pack>/policy.pt` (git-ignored).

## Training (`train.py`)

1. **Imitation** (`imitation=300` sessions): cross-entropy towards the greedy policy's choices.
2. **Actor-critic** (`episodes=3000`, batches of `batch=16` episodes, Adam `lr=0.003`):
   ```
   G_t      = outcome - question_cost × (questions asked from decision t on)
   A_t      = normalise(G_t - V(s_t))           # per batch
   loss     = -mean(A_t · log π(a_t|s_t)) + 0.5 · mean((V(s_t) - G_t)²) - 0.01 · mean(entropy)
   ```
   outcome = +1 right / −0.25 uncertain / −3 wrong; question_cost = 0.02.
3. **Checkpoints**: every `eval_every=250` episodes the arg-max policy is scored on the
   validation cases (`--val-size`, default 1500, with `--data`) and the best one is kept.

CLI: `verity train <pack> [--data DIR] [--mock-llm] [--episodes] [--batch] [--imitation]
[--eval-every] [--val-size] [--lr] [--seed] [--out]`.

## Evaluation

`verity evaluate <pack> [--data DIR] [--split test]` runs every available policy on the same
people. Use at least a few thousand test people (`verity data generate cv --size 20000
--cases-only --out datasets/cv-large`): with 450 the mean reward is only accurate to about
±0.06, which is larger than the differences between policies (see modeling.md §11).
A new policy is an improvement only if its **reward** is higher on the test split.

## Results

On the CV pack (3,000 test people, keyword grading) the learned policy does **not** beat greedy:
greedy +0.261, learned +0.199 / +0.232 / +0.228 for seeds 0 / 1 / 2. Greedy stays the default.
Why, and when RL is expected to pay off: [../modeling.md](../modeling.md) §13.
