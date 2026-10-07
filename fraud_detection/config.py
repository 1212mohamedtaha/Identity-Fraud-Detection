"""Paths and constants shared across the package."""
import os
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("FRAUD_DATA_ROOT", PROJECT_ROOT / "data"))
CHECKPOINT_DIR = Path(os.environ.get(
    "FRAUD_CHECKPOINT_DIR",
    DATA_ROOT / "checkpoints" / "ghrl" / "RL" / "epoch_201_success_0.85_turn_9.71953125",
))
GRAPHS_FILE = DATA_ROOT / "preprocessed_graphs" / "test.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Terminal actions are appended to the tail of every action space.
FRAUD = -2
NON_FRAUD = -1

EPS = 1e-6
EDGE_PAD = (0, 0, 0)

# Dialogue limits.
MAX_EXPLORING_TIME_STEP = 40
MAX_WORKER_EXPLORING_TIME_STEP = 10
MIN_WORKER_QA_TURN = 3

# Question generation.
NEGATIVE_SAMPLED_ANSWER_NUM = 2
OPTIONS = ("A", "B", "C", "D")
NOT_SURE_TEXT = "Not Sure"

# Node features.
PERSONAL_NODE_FIELD = 4
ONE_STEP_NODE_FIELD = 1
SE_FREQS_FIELD = 10
DEGREE_FIELD = 10
STATIC_FEATURE_SIZE = PERSONAL_NODE_FIELD + ONE_STEP_NODE_FIELD + SE_FREQS_FIELD + DEGREE_FIELD
DYNAMIC_FEATURE_SIZE = 7
INIT_NODE_FEATURE_SIZE = STATIC_FEATURE_SIZE + DYNAMIC_FEATURE_SIZE

# Hidden sizes the checkpoint was trained with.
GNN_LAYER_SIZES = (40, 50)
MANAGER_AGG_SIZE = 100
MANAGER_STATE_REST = (
    4 + 2 * 4
    + (MAX_WORKER_EXPLORING_TIME_STEP + 1) * 3 * 4
    + (MAX_EXPLORING_TIME_STEP + 1)
)
WORKERS_STATE_REST = (MAX_WORKER_EXPLORING_TIME_STEP + 1) * 5
