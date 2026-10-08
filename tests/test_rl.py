from verity.core.engine import Session
from verity.rl.policy import load_learned_policy
from verity.rl.train import train


def test_training_saves_a_policy_that_runs(toy, tmp_path):
    out = tmp_path / "policy.pt"
    train(toy, episodes=50, out=out, log=None)
    assert out.exists()
    session = Session(toy, policy=load_learned_policy(out))
    while not session.finished:
        session.answer("A")
    assert session.verdict is not None


def test_learned_policy_becomes_available_after_training(toy):
    assert "learned" not in toy.policy_names()
    train(toy, episodes=20, out=toy.learned_policy_path(), log=None)
    assert "learned" in toy.policy_names()
    assert toy.policy("learned").name == "learned"
