"""Simulated people and evaluation: run many sessions where the truth is known."""
import random

from .types import REFUTED, SUPPORTED, UNCERTAIN

NOT_SURE = "I'm not sure."

# Reward for the final verdict, used by training and by `verity evaluate`.
# "Uncertain" is mildly negative: ending without an answer is a (small) failure.
VERDICT_REWARD = {"correct": 1.0, "uncertain": -0.25, "wrong": -1.0}
QUESTION_COST = 0.02     # subtracted per question asked


def episode_reward(session, truth, question_cost=QUESTION_COST):
    status = session.verdict.status
    if status == UNCERTAIN:
        outcome = VERDICT_REWARD["uncertain"]
    elif status == expected_status(truth):
        outcome = VERDICT_REWARD["correct"]
    else:
        outcome = VERDICT_REWARD["wrong"]
    return outcome - question_cost * len(session.history)


class StatisticalRespondent:
    """Answers each probe correctly with the probe's own pass rate.

    ``truth`` maps claim id -> bool. Every simulated person also gets a personal
    knowledge offset (``spread``), so some genuine people are sharper than others.
    """

    def __init__(self, truth, rng, spread=0.1):
        self.truth = truth
        self.rng = rng
        self.offset = rng.uniform(-spread, spread)

    def passes(self, probe):
        rate = probe.p_true if self.truth.get(probe.claim_id, True) else probe.p_false
        rate = min(max(rate + self.offset, 0.0), 1.0)
        return self.rng.random() < rate

    def answer(self, probe):
        if probe.is_free_text:
            return probe.answer if self.passes(probe) else NOT_SURE
        if self.passes(probe):
            return probe.answer
        wrong = [c.id for c in probe.choices if c.id != probe.answer]
        return self.rng.choice(wrong)


def expected_status(truth):
    return SUPPORTED if all(truth.values()) else REFUTED


def run_episode(pack, policy, rng):
    """Play one simulated session. Returns (session, truth)."""
    from .engine import Session

    inputs, truth = pack.sample_case(rng)
    session = Session(pack, inputs, policy=policy)
    respondent = pack.respondent(truth, rng)
    while not session.finished:
        session.answer(respondent.answer(session.current))
    return session, truth


def evaluate(pack, policy_name, episodes=200, seed=0):
    """Run ``episodes`` simulated sessions and summarise how well the policy did.

    ``reward`` is the average of the score RL training maximises (see episode_reward).
    """
    rng = random.Random(seed)
    correct = uncertain = wrong = questions = 0
    total_reward = 0.0
    for _ in range(episodes):
        policy = pack.policy(policy_name, rng=rng)
        session, truth = run_episode(pack, policy, rng)
        status = session.verdict.status
        questions += len(session.history)
        total_reward += episode_reward(session, truth)
        if status == UNCERTAIN:
            uncertain += 1
        elif status == expected_status(truth):
            correct += 1
        else:
            wrong += 1
    return {
        "policy": policy_name,
        "episodes": episodes,
        "accuracy": correct / episodes,
        "uncertain": uncertain / episodes,
        "wrong": wrong / episodes,
        "avg_questions": questions / episodes,
        "reward": total_reward / episodes,
    }
