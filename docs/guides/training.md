# Guide: train and evaluate a questioning policy

This guide trains the reinforcement-learning (RL) policy that decides **which question to
ask next and when to stop**. Everything runs on a normal laptop CPU in under a minute per
pack, with no API key and no data download.

## What you are training

During a session the belief model keeps a probability for each claim. The *policy* looks
at those probabilities and the remaining questions, then picks one question or stops.
Three policies exist out of the box:

| Policy | How it chooses |
| --- | --- |
| `random` | Any open question. A baseline to beat. |
| `greedy` | Digs into the weakest claim, asking its most informative question. Stops once the verdict is settled. |
| `learned` | A small neural network trained here with RL. |
| `legacy` | Identity pack only: the original project's pretrained GNN + hierarchical RL model. |

## How training works (in plain words)

1. **Invent a person.** The pack's `sample_case` creates someone whose truth we know,
   e.g. "really works at X, but lied about the university".
2. **Play a full session.** A simulated respondent answers each question: someone with a
   true claim passes a question with probability `p_true`, someone lying with `p_false`.
3. **Score the session.** +1 for a correct verdict, −3 for a wrong one, −0.25 if it ended
   "uncertain", and −0.02 for every question asked (shorter is better).
4. **Learn.** First the network copies the greedy policy (imitation), then it improves by
   actor-critic: choices that led to better-than-predicted results become more likely.
   The best version on validation people is kept.

## Step by step

```bash
# 0. once: install
pip install -e ".[dev]"

# 1. see how the built-in policies do (300 simulated people, same for every policy)
verity evaluate identity

# 2. train (2000 simulated sessions, ~25 s); prints progress every 200 episodes
verity train identity

# 3. compare again; "learned" now appears in the table
verity evaluate identity

# 4. try it yourself
verity play identity --policy learned
verity serve          # then choose "learned" under "Question strategy"
```

The trained model is saved to `models/identity/policy.pt` (in the folder you run the
command from). Delete that file to go back to the built-in policies only.

## Reading the results

```
policy      accuracy  uncertain   wrong  questions  reward
greedy         61.7%      28.3%   10.0%        9.2  +0.262
random         66.7%      24.0%    9.3%       14.2  +0.230
learned        65.7%      28.0%    6.3%       10.6  +0.311
legacy         48.7%      48.3%    3.0%       10.2  +0.131
```

- **accuracy**: verdict matched the truth.
- **uncertain**: the session ended without a verdict (not enough evidence).
- **wrong**: confident but wrong. The worst outcome.
- **questions**: average questions per session. Fewer is a better experience.
- **reward**: the single number training maximises (combines all of the above).
  **Compare policies by reward.**

Here the learned policy beats greedy on reward: it is wrong less often for about one
more question. `legacy` is cautious; it stops on its own decision, which often leaves
claims "uncertain" for the belief model.

The same for the CV pack:

```bash
verity train cv && verity evaluate cv
```

## Knobs

| Option | Default | Effect |
| --- | --- | --- |
| `--episodes` | 2000 | More episodes = slower but usually steadier results. |
| `--question-cost` | 0.02 | Higher makes the policy ask fewer questions (and decide less often). |
| `--lr` | 0.01 | Learning rate. Lower it (e.g. 0.003) with more episodes if results jump around. |
| `--seed` | 0 | Change it to check the result is not luck. |
| `--out` | `models/<pack>/policy.pt` | Where to save. |

`verity evaluate` also takes `--episodes`, `--seed` and `--policies greedy learned`.

## Doing it from Python

```python
from verity.packs import get_pack
from verity.rl.train import train
from verity.core.simulation import evaluate

pack = get_pack("cv")
train(pack, episodes=3000, out=pack.learned_policy_path())
print(evaluate(pack, "learned", episodes=500))
```

## Training on a synthetic dataset (CV pack)

```bash
# 1. data: a small set with answers (for fitting) and a large set (for training and testing)
verity data generate cv --size 3000                                   # -> datasets/cv/
verity data generate cv --size 20000 --seed 7 --cases-only --out datasets/cv-large

# 2. measure graders and fit the model (question difficulty, fatigue, person spread, priors)
verity data grader-eval cv --data datasets/cv
verity data fit cv --data datasets/cv

# 3. baseline on 3,000 held-out test people
verity evaluate cv --data datasets/cv-large --policies greedy random

# 4. train (imitation warm start + actor-critic, ~10 min per seed on a laptop CPU)
verity train cv --data datasets/cv-large --episodes 6000 --eval-every 500

# 5. compare on the test split
verity evaluate cv --data datasets/cv-large --policies greedy learned
```

Add `--mock-llm` to steps 3–5 to run the LLM code path with the mock LLM (grading closer to
what a real LLM would do). Train with several `--seed` values and keep the result only if it beats
greedy on the **test** split. The maths behind every step: [../modeling.md](../modeling.md).

What the dataset contains and its limits: [../specs/dataset.md](../specs/dataset.md).

## Making training more realistic

The simulator is only as good as its assumptions (`p_true`, `p_false` per question). To
make the learned policy worth more than greedy:

1. Collect real sessions (with consent) and estimate each question's real pass rates.
2. Put those rates on the probes (`ProbeGenerator`) and, if needed, write a richer
   `Respondent` for your pack (e.g. people get tired, related topics are known together,
   some answers can be looked up).
3. Retrain and compare with `verity evaluate`. Keep the new policy only if its reward is higher.

Details of features, network and loss: [../specs/rl.md](../specs/rl.md).
