# Verity

**Verify claims with a short, adaptive conversation.**

Someone claims something: *"I work at X"*, *"I know SQL"*. Verity builds the knowledge
that backs the claim, asks the most useful questions, grades the answers and keeps a
probability for every claim. It stops as soon as it is sure, and explains its verdict.

The core knows nothing about any domain. Each use case is a **domain pack**:

| Pack | What it checks | Questions | Grading |
| --- | --- | --- | --- |
| `identity` | Where you work, studied, live, were born | Multiple choice about nearby places | Exact match |
| `cv` | The skills and levels your CV claims ("senior Python") | Free-text interview questions | LLM rubric grading (keywords offline) |

Both run **with no API key** (offline mode). With an LLM configured, the CV pack reads
the CV, writes questions about *your* projects and grades answers properly.

---

## 1. Install (once)

You need Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## 2. Run the web app

```bash
verity serve
```

Open http://127.0.0.1:8000, pick a pack and answer the questions.
Interactive API docs: http://127.0.0.1:8000/docs.

## 3. Or play in the terminal

```bash
verity play identity
verity play cv --input cv=@my_cv.txt --input job=@job.txt
```

`--input key=value` sets a pack input; `key=@file` reads it from a file.

## 4. Use an LLM (optional)

Claude is the default; any OpenAI-compatible API works too.

```bash
# Claude
export ANTHROPIC_API_KEY=sk-ant-...          # uses claude-opus-5-5 by default
export VERITY_LLM_MODEL=claude-sonnet-5-5    # optional: pick another model

# Anything OpenAI-compatible: OpenAI, Ollama (local, free), LM Studio, vLLM, Groq...
export VERITY_LLM_PROVIDER=openai
export VERITY_LLM_BASE_URL=http://localhost:11434/v1     # Ollama example
export VERITY_LLM_MODEL=llama3.1

# Force offline mode
export VERITY_LLM_PROVIDER=none
```

On Windows use `set NAME=value` instead of `export`. See [.env.example](.env.example) and
[docs/guides/llm-providers.md](docs/guides/llm-providers.md).

## 5. Train and evaluate the question policy (RL)

A *policy* decides which question to ask next and when to stop. Compare policies on
simulated people whose truth is known, then train your own:

```bash
verity evaluate identity          # random vs greedy vs the original RL model
verity train identity             # ~2 min on a laptop CPU -> models/identity/policy.pt
verity evaluate identity          # now also lists "learned"
```

Use it in the web app ("Question strategy") or with `verity play identity --policy learned`.
Step-by-step guide: [docs/guides/training.md](docs/guides/training.md).

## 6. Test

```bash
pytest            # ~10 s, fully offline
ruff check .      # lint
```

---

## How it works

```
inputs ─► ClaimExtractor ─► claims ─► KnowledgeSource ─► graph ─► ProbeGenerator ─► probes
                                                                                      │
          ┌───────────────────────────────────────────────────────────────────────────┘
          ▼
   Policy picks a probe ─► person answers ─► Assessor grades ─► BeliefModel updates
          ▲                                                            │
          └────────────── until sure, or out of questions ◄────────────┘
                                          │
                                          ▼
                              Verdict per claim + explanation
```

Start with [docs/architecture.md](docs/architecture.md), then the specs in
[docs/specs/](docs/specs/). To build your own use case, follow
[docs/guides/adding-a-pack.md](docs/guides/adding-a-pack.md).

## Project layout

```
verity/
  core/        building blocks, engine, belief model, policies, simulation
  llm/         provider-agnostic LLM layer (Claude, OpenAI-compatible, fake for tests)
  rl/          policy network, features, REINFORCE training
  packs/
    identity/  identity pack + the original GNN / hierarchical RL model (legacy/) and its data
    cv/        CV skills pack: the reference example of extending the core
  api/         FastAPI app
  web/         chat UI (plain HTML, CSS and JavaScript)
  cli.py       the `verity` command
docs/          architecture, specs, guides, decision records
tests/         offline tests (tests/toy_pack.py is the smallest possible pack)
AGENTS.md      rules and recipes for AI coding agents (and humans)
```

## Responsible use

Verity is a research and practice tool. Its verdicts are probabilities from a model
tuned on simulated people, not facts. Don't use them on their own to deny anyone a
job, a loan or a service, and get consent before checking claims about real people.
