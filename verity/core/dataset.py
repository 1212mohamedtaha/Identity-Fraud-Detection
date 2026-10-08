"""Datasets of simulated people ("cases") saved as JSON Lines, split train / val / test.

A case is one person: ``{"id", "split", "inputs", "truth", ...}`` where ``inputs`` start a
session and ``truth`` maps claim id -> real level (or bool for yes/no claims). Packs may
add more fields (e.g. personality traits). See docs/specs/dataset.md.
"""
import json
import random
from pathlib import Path

SPLITS = (("train", 0.70), ("val", 0.15), ("test", 0.15))


def split_for(index, size):
    """Deterministic split: the first 70% of cases are train, then 15% val, 15% test."""
    position, start = index / size, 0.0
    for name, share in SPLITS:
        start += share
        if position < start:
            return name
    return SPLITS[-1][0]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_dataset(pack, size, seed, out):
    """Create ``size`` cases with the pack's ``make_case`` and write them to ``out/cases.jsonl``.
    The pack may write extra files (``write_dataset_extras``). Returns the cases."""
    rng = random.Random(seed)
    cases = []
    for index in range(size):
        case = pack.make_case(rng, index)
        case["split"] = split_for(index, size)
        cases.append(case)
    out = Path(out)
    write_jsonl(out / "cases.jsonl", cases)
    pack.write_dataset_extras(cases, out, random.Random(seed + 1))
    (out / "info.json").write_text(json.dumps(
        {"pack": pack.name, "size": size, "seed": seed, "splits": dict(SPLITS)}, indent=2))
    return cases


def load_cases(data_dir, split=None):
    cases = read_jsonl(Path(data_dir) / "cases.jsonl")
    return [c for c in cases if split is None or c["split"] == split]
