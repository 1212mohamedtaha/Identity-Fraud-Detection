"""A small neural policy: scores every open probe plus a "stop" option, and predicts how
the session will end (the value, used only during training)."""
from pathlib import Path

import torch
import torch.nn as nn

from ..core.interfaces import Policy
from ..core.policies import open_candidates, verdict_settled
from .features import PROBE_FEATURES, STATE_FEATURES, probe_features, state_features


def mlp(inputs, hidden):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, 1))


class PolicyNet(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.probe_head = mlp(PROBE_FEATURES, hidden)   # one score per candidate probe
        self.stop_head = mlp(STATE_FEATURES, hidden)    # score of stopping now
        self.value_head = mlp(STATE_FEATURES, hidden)   # expected reward from here (critic)

    def forward(self, probe_feats, state_feats):
        """probe_feats: (n, PROBE_FEATURES), state_feats: (STATE_FEATURES,) -> logits (n + 1,)"""
        probe_scores = self.probe_head(probe_feats).squeeze(-1)
        return torch.cat([probe_scores, self.stop_head(state_feats)])

    def value(self, state_feats):
        return self.value_head(state_feats).squeeze(-1)


def encode(state, candidates):
    gains = [state.belief.expected_gain(p) for p in candidates]
    probe_feats = torch.tensor([probe_features(state, p, g) for p, g in zip(candidates, gains)],
                               dtype=torch.float32)
    state_feats = torch.tensor(state_features(state, gains), dtype=torch.float32)
    return probe_feats, state_feats


class LearnedPolicy(Policy):
    """Uses a PolicyNet.

    - ``explore=True``: samples actions and records, per decision, the log-probability,
      entropy, predicted value and number of questions asked so far (for training).
    - ``imitate=other_policy``: asks ``other_policy`` and follows it, recording the network's
      log-probability of that same choice (for the imitation warm start).
    - otherwise: always takes the best-scored action.
    """

    name = "learned"

    def __init__(self, net, explore=False, imitate=None):
        self.net = net
        self.explore = explore
        self.imitate = imitate
        self.steps = []          # one dict per decision made by the network

    def reset(self, state):
        self.steps = []
        if self.imitate:
            self.imitate.reset(state)

    def observe(self, state, probe, observation):
        if self.imitate:
            self.imitate.observe(state, probe, observation)

    def choose(self, state):
        candidates = open_candidates(state)     # unasked probes of undecided claims
        if not candidates or verdict_settled(state):
            return None
        probe_feats, state_feats = encode(state, candidates)

        if self.imitate:
            target = self.imitate.choose(state)
            index = len(candidates) if target is None else candidates.index(target)
            logits = self.net(probe_feats, state_feats)
            self.steps.append({"log_prob": torch.log_softmax(logits, 0)[index]})
        elif self.explore:
            logits = self.net(probe_feats, state_feats)
            dist = torch.distributions.Categorical(logits=logits)
            choice = dist.sample()
            self.steps.append({"log_prob": dist.log_prob(choice), "entropy": dist.entropy(),
                               "value": self.net.value(state_feats), "asked": len(state.history)})
            index = int(choice)
        else:
            with torch.no_grad():
                index = int(self.net(probe_feats, state_feats).argmax())

        return None if index == len(candidates) else candidates[index]


def save_policy(net, path, info=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "info": info or {}}, path)


def load_learned_policy(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    net = PolicyNet()
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    return LearnedPolicy(net)
