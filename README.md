# Identity Fraud Detection

An interactive identity-verification chat. A hierarchical reinforcement-learning agent walks a
per-applicant knowledge graph (university, company, birthplace, home address and the places around
them) and asks multiple-choice questions about it. Depending on how well the applicant answers, it
decides **verified** or **flagged for review** after about ten questions.

> Research prototype. Decisions that depend on where someone lives, studied or was born can encode
> bias; don't use this on real applicants without an evaluation of fairness and accuracy.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m fraud_detection            # http://127.0.0.1:5000  (--host, --port, --debug)
```

Sessions are kept in memory, so run a single worker process.
Set `SECRET_KEY` to keep browser sessions valid across restarts.

## How it works

| Module | Role |
| --- | --- |
| `fraud_detection/models.py` | GNN, state trackers, manager and workers (PyTorch) and checkpoint loading |
| `fraud_detection/graph.py` | Pads a raw graph into network inputs |
| `fraud_detection/features.py` | Static and per-turn node features |
| `fraud_detection/dialogue.py` | Dialogue state and the rules for updating it after each action/answer |
| `fraud_detection/policy.py` | Runs the trained policy (greedy) to pick the next action |
| `fraud_detection/language.py` | Question templates and answer options |
| `fraud_detection/session.py` | `DialogueSession` (one applicant) and `Engine` (models loaded once) |
| `fraud_detection/web/` | Flask app and the chat UI (`templates/`, `static/`) |

The manager decides which personal detail to explore next; the worker for that detail chooses which
nearby place to ask about. A correct answer marks the node as known; a wrong answer or "Not Sure"
as unknown. Either level can end with a fraud / non-fraud decision.

```python
from fraud_detection import Engine

dialogue = Engine().new_session()
dialogue.current          # Turn(status="question", question=..., choices=[{"id": "A", ...}, ...])
dialogue.answer("A")      # -> next Turn; status becomes "fraud" or "non_fraud" at the end
```

### HTTP API

| Method & path | Purpose |
| --- | --- |
| `POST /api/dialogue` | Start (or restart) a dialogue; returns `{turn, history}` |
| `GET /api/dialogue` | Resume the current dialogue (`204` if none) |
| `POST /api/dialogue/answer` | Body `{"choice": "A"}`; returns the next `{turn, history}` |

## Data and training artefacts

- `data/preprocessed_graphs/` – applicant graphs; the app uses `test.json`.
- `data/checkpoints/` – pretrained weights, loaded with `torch.load(..., weights_only=True)`.
- `notebooks/ConstructGraph.ipynb` – builds a graph for a new applicant with Google Places
  (needs your own API key).
- Paths can be overridden with `FRAUD_DATA_ROOT` and `FRAUD_CHECKPOINT_DIR`.

The original single-notebook implementation (`main.ipynb`) is in the git history (commit `29679fb`).
Training-only code that the app never used (reward computation, flat-RL variant, rule-based warm-up)
was not carried over.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```
