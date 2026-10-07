import random

import pytest

pytest.importorskip("torch")

from fraud_detection import Engine  # noqa: E402
from fraud_detection.policy import Policy  # noqa: E402


@pytest.fixture(scope="session")
def policy():
    return Policy()


@pytest.fixture
def engine(policy):
    return Engine(policy=policy, rng=random.Random(0))
