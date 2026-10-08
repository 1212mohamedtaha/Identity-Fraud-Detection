"""A small neural policy: scores every open probe plus a "stop" option."""
from pathlib import Path

import torch
import torch.nn as nn

from ..core.interfaces import Policy
from ..core.policies import open_candidates, verdict_settled
from .features import PROBE_FEATURES, STOP_FEATURES, probe_features, stop_features


class PolicyNet(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.probe_head = nn.Sequential(nn.Linear(PROBE_FEATURES, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.stop_head = nn.Sequential(nn.Linear(STOP_FEATURES, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, probe_feats, stop_feats):
        """probe_feats: (n, PROBE_FEATURES), stop_feats: (STOP_FEATURES,) -> logits (n + 1,)"""
        probe_scores = self.probe_head(probe_feats).squeeze(-1)
        stop_score = self.stop_head(stop_feats)
        return torch.cat([probe_scores, stop_score])


class LearnedPolicy(Policy):
    """Uses a PolicyNet. With ``explore=True`` it samples actions and records their
    log-probabilities for training; otherwise it always takes the best-scored action."""

    name = "learned"

    def __init__(self, net, explore=False):
        self.net = net
        self.explore = explore
        self.log_probs = []    # filled while exploring, used by training
        self.entropies = []

    def reset(self, state):
        self.log_probs = []
        self.entropies = []

    def choose(self, state):
        candidates = open_candidates(state)     # unasked probes of undecided claims
        if not candidates or verdict_settled(state):
            return None
        probe_feats = torch.tensor([probe_features(state, p) for p in candidates], dtype=torch.float32)
        stop_feats = torch.tensor(stop_features(state), dtype=torch.float32)

        if self.explore:
            logits = self.net(probe_feats, stop_feats)
            dist = torch.distributions.Categorical(logits=logits)
            index = dist.sample()
            self.log_probs.append(dist.log_prob(index))
            self.entropies.append(dist.entropy())
            index = int(index)
        else:
            with torch.no_grad():
                index = int(self.net(probe_feats, stop_feats).argmax())

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
