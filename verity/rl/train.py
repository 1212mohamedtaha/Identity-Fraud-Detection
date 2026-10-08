"""Train a LearnedPolicy with REINFORCE (the simplest policy-gradient method).

Each episode: simulate a person whose truth we know, run a full session with the
exploring policy, then reward it (see verity.core.simulation.episode_reward):

    +1 if the verdict is right, -0.25 if it ends "uncertain", -1 if it is wrong,
    minus ``question_cost`` for every question asked.

The policy is nudged towards the choices made in episodes that scored above average.
A small entropy bonus keeps it exploring, so it does not settle too early on a habit
such as "always stop at once".
"""
import random

import torch

from ..core.simulation import QUESTION_COST, episode_reward, run_episode
from .policy import LearnedPolicy, PolicyNet, save_policy


def train(pack, episodes=2000, lr=0.01, question_cost=QUESTION_COST, entropy_bonus=0.01,
          seed=0, out=None, log_every=200, log=print, cases=None):
    """Train a policy. With ``cases`` (a dataset's train split) each episode plays a random
    case from it; otherwise every episode makes a fresh simulated person."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    net = PolicyNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    baseline = 0.0       # running average reward; rewards above it are "good"
    recent = []

    for episode in range(1, episodes + 1):
        policy = LearnedPolicy(net, explore=True)
        case = rng.choice(cases) if cases else None
        session, truth = run_episode(pack, policy, rng, case)
        reward = episode_reward(session, truth, question_cost)

        if policy.log_probs:
            advantage = reward - baseline
            loss = -advantage * torch.stack(policy.log_probs).sum()
            loss = loss - entropy_bonus * torch.stack(policy.entropies).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        baseline = reward if episode == 1 else 0.95 * baseline + 0.05 * reward

        recent.append((reward, len(session.history)))
        if log and episode % log_every == 0:
            avg_reward = sum(r for r, _ in recent) / len(recent)
            avg_questions = sum(q for _, q in recent) / len(recent)
            log(f"episode {episode:5d}  avg reward {avg_reward:+.3f}  avg questions {avg_questions:.1f}")
            recent = []

    if out:
        save_policy(net, out, {"pack": pack.name, "episodes": episodes, "question_cost": question_cost})
    return net
