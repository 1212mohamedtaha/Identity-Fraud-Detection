import numpy as np

from verity.core.engine import Session
from verity.packs.identity import IdentityPack, graphs
from verity.packs.identity.legacy.features import static_features


def test_claims_and_probes_come_from_the_graph():
    session = Session(IdentityPack())
    kinds = {c.id for c in session.state.claims}
    assert kinds == {"company", "university", "live_in", "born_in"}
    raw = graphs()[0]
    assert len(session.state.probes) == sum(len(v) for v in raw["one_step_nodes"].values())
    probe = session.current
    assert probe.answer in {c.id for c in probe.choices}
    assert probe.choices[-1].text == "Not Sure"


def test_static_features_match_the_precomputed_dataset():
    raw = graphs()[0]
    assert np.allclose(static_features(raw), raw["static_feature"])


def test_genuine_person_is_supported_with_greedy():
    session = Session(IdentityPack())
    while not session.finished:
        session.answer(session.current.answer)
    assert session.verdict.status == "supported"


def test_legacy_rl_model_runs_as_a_policy():
    pack = IdentityPack()
    honest = Session(pack, policy=pack.policy("legacy"))
    while not honest.finished:
        honest.answer(honest.current.answer)
    assert honest.verdict.notes == ["The original RL model decided: not fraud."]

    impostor = Session(pack, policy=pack.policy("legacy"))
    while not impostor.finished:
        impostor.answer("D")          # always "Not Sure"
    assert impostor.verdict.notes == ["The original RL model decided: fraud."]
    assert impostor.verdict.status == "refuted"
