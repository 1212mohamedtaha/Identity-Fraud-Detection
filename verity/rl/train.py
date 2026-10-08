"""Train a LearnedPolicy: imitation warm start, then actor-critic.

1. **Imitation.** Play sessions with the greedy policy and teach the network to make the same
   choices (cross-entropy). RL then starts from a sensible policy instead of a random one.
2. **Actor-critic.** Play sessions with the exploring network. Reward (see
   verity.core.simulation.episode_reward): +1 right verdict, -0.25 "uncertain", -3 wrong,
   and -``question_cost`` for every question, charged when the question is asked.
   For each decision, the *return* is everything earned from that decision on; the critic
   (value head) predicts it, and the actor is pushed towards decisions whose return beat
   the prediction (the *advantage*). Updates use batches of episodes; advantages are
   normalised per batch; a small entropy bonus keeps exploring.
3. **Checkpoints.** Every ``eval_every`` episodes the greedy-argmax version of the network
   is scored on ``val_cases`` (or fresh simulated people); the best one is kept.
"""
import copy
import random

import torch

from ..core.policies import GreedyPolicy
from ..core.simulation import QUESTION_COST, VERDICT_REWARD, evaluate, expected_status, run_episode
from ..core.types import UNCERTAIN
from .policy import LearnedPolicy, PolicyNet, save_policy


def outcome_reward(session, truth):
    status = session.verdict.status
    if status == UNCERTAIN:
        return VERDICT_REWARD["uncertain"]
    if status == expected_status(truth, session.state.claims):
        return VERDICT_REWARD["correct"]
    return VERDICT_REWARD["wrong"]


def returns_for(policy, session, truth, question_cost):
    """Return-to-go for each decision: the outcome minus the cost of the questions still to come
    (the decision's own question included)."""
    outcome = outcome_reward(session, truth)
    total = len(session.history)
    return [outcome - question_cost * (total - step["asked"]) for step in policy.steps]


def score(pack, net, cases, seed, episodes):
    """Average reward of the network's best-scored actions."""
    net.eval()
    result = evaluate(pack, lambda rng: LearnedPolicy(net), episodes=episodes, seed=seed, cases=cases)
    net.train()
    return result["reward"]


def train(pack, episodes=3000, lr=0.003, batch=16, imitation=300, entropy_bonus=0.01, value_weight=0.5,
          question_cost=QUESTION_COST, seed=0, out=None, cases=None, val_cases=None, eval_every=250,
          eval_episodes=200, log=print):
    """Train a policy; returns the best network. With ``cases`` (a dataset's train split) each
    episode plays a random case from it; otherwise fresh simulated people."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)     # the network is tiny: one thread is fastest, and several runs can share a CPU
    rng = random.Random(seed)
    net = PolicyNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    def play(policy):
        case = rng.choice(cases) if cases else None
        return run_episode(pack, policy, rng, case)

    # 1. imitation warm start
    losses = []
    for episode in range(1, imitation + 1):
        policy = LearnedPolicy(net, imitate=GreedyPolicy())
        play(policy)
        losses += [-step["log_prob"] for step in policy.steps]
        if episode % batch == 0 and losses:
            optimizer.zero_grad()
            torch.stack(losses).mean().backward()
            optimizer.step()
            losses = []
    if log and imitation:
        log(f"imitation: {imitation} greedy sessions copied")

    # 2. actor-critic with 3. validation checkpoints
    best_net, best_score = copy.deepcopy(net), score(pack, net, val_cases, seed, eval_episodes)
    if log:
        log(f"episode     0  validation reward {best_score:+.3f}")
    batch_steps, batch_returns, recent = [], [], []
    for episode in range(1, episodes + 1):
        policy = LearnedPolicy(net, explore=True)
        session, truth = play(policy)
        batch_steps += policy.steps
        batch_returns += returns_for(policy, session, truth, question_cost)
        recent.append(outcome_reward(session, truth) - question_cost * len(session.history))

        if episode % batch == 0 and batch_steps:
            returns = torch.tensor(batch_returns)
            values = torch.stack([s["value"] for s in batch_steps])
            advantages = returns - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
            actor = -(advantages * torch.stack([s["log_prob"] for s in batch_steps])).mean()
            critic = ((values - returns) ** 2).mean()
            explore = torch.stack([s["entropy"] for s in batch_steps]).mean()
            optimizer.zero_grad()
            (actor + value_weight * critic - entropy_bonus * explore).backward()
            optimizer.step()
            batch_steps, batch_returns = [], []

        if episode % eval_every == 0:
            current = score(pack, net, val_cases, seed, eval_episodes)
            if current > best_score:
                best_net, best_score = copy.deepcopy(net), current
            if log:
                log(f"episode {episode:5d}  training reward {sum(recent) / len(recent):+.3f}  "
                    f"validation reward {current:+.3f}  (best {best_score:+.3f})")
            recent = []

    best_net.eval()
    if out:
        save_policy(best_net, out, {"pack": pack.name, "episodes": episodes, "seed": seed,
                                    "validation_reward": best_score})
    return best_net
