"""Simulated people and evaluation: run many sessions where the truth is known."""
import random

from .types import REFUTED, SUPPORTED, UNCERTAIN

NOT_SURE = "I'm not sure."

# Reward for the final verdict, used by training and by `verity evaluate`.
# A wrong verdict (e.g. telling a real senior "you are not senior") costs three times what a
# correct one earns; ending "not sure yet" is a small failure. See docs/modeling.md.
VERDICT_REWARD = {"correct": 1.0, "uncertain": -0.25, "wrong": -3.0}
QUESTION_COST = 0.02     # subtracted per question asked


def episode_reward(session, truth, question_cost=QUESTION_COST):
    """``truth`` maps claim id -> real level (see true_levels)."""
    status = session.verdict.status
    if status == UNCERTAIN:
        outcome = VERDICT_REWARD["uncertain"]
    elif status == expected_status(truth, session.state.claims):
        outcome = VERDICT_REWARD["correct"]
    else:
        outcome = VERDICT_REWARD["wrong"]
    return outcome - question_cost * len(session.history)


def true_levels(truth, claims):
    """Turn a pack's truth into real levels: claim id -> level index.

    A pack may give a level index per claim, or (for yes/no use cases) a bool:
    True means "exactly the claimed level", False "one level below it".
    """
    levels = {}
    for claim in claims:
        value = truth.get(claim.id, False)       # a claim the truth does not mention is false
        if isinstance(value, bool):
            value = claim.claimed_level if value else max(claim.claimed_level - 1, 0)
        levels[claim.id] = value
    return levels


class StatisticalRespondent:
    """Passes each probe with the probe's own pass rate at the person's real level.

    ``truth`` maps claim id -> real level. Every simulated person also gets a personal
    offset (``spread``), so some people do a bit better or worse than their level suggests.
    """

    def __init__(self, truth, rng, spread=0.1):
        self.truth = truth
        self.rng = rng
        self.offset = rng.uniform(-spread, spread)

    def passes(self, probe):
        level = self.truth[probe.claim_id]
        rate = min(max(probe.pass_rates[level] + self.offset, 0.0), 1.0)
        return self.rng.random() < rate

    def answer(self, probe):
        if probe.is_free_text:
            return probe.answer if self.passes(probe) else NOT_SURE
        if self.passes(probe):
            return probe.answer
        wrong = [c.id for c in probe.choices if c.id != probe.answer]
        return self.rng.choice(wrong)


def expected_status(truth, claims):
    """The right verdict: supported when every claim holds at the person's real level."""
    return SUPPORTED if all(c.holds_at(truth[c.id]) for c in claims) else REFUTED


def run_episode(pack, policy, rng, case=None):
    """Play one simulated session with ``case`` (from a dataset) or a freshly made one.
    Returns (session, truth as real levels)."""
    from .engine import Session

    case = case or pack.make_case(rng, 0)
    session = Session(pack, case["inputs"], policy=policy)
    truth = true_levels(case["truth"], session.state.claims)
    respondent = pack.respondent(truth, rng, case)
    while not session.finished:
        session.answer(respondent.answer(session.current))
    return session, truth


def evaluate(pack, policy_name, episodes=200, seed=0, cases=None):
    """Run simulated sessions and summarise how well the policy did.

    With ``cases`` (e.g. a dataset's test split) every case is played once; otherwise
    ``episodes`` fresh people are made. ``reward`` is the average of the score RL
    training maximises (see episode_reward).
    """
    rng = random.Random(seed)
    correct = uncertain = wrong = questions = 0
    total_reward = 0.0
    plan = cases if cases is not None else [None] * episodes
    episodes = len(plan)
    for case in plan:
        policy = pack.policy(policy_name, rng=rng)
        session, truth = run_episode(pack, policy, rng, case)
        status = session.verdict.status
        questions += len(session.history)
        total_reward += episode_reward(session, truth)
        if status == UNCERTAIN:
            uncertain += 1
        elif status == expected_status(truth, session.state.claims):
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
