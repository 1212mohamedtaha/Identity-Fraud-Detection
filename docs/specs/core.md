# Spec: core

Code: `verity/core/`. The core must stay domain-agnostic (no identity, CV or other domain words).

## Data types (`types.py`)

| Type | Fields | Notes |
| --- | --- | --- |
| `Claim` | `id`, `text`, `kind`, `data` | `id` is unique within a session. `data` is free space for the pack. |
| `Choice` | `id`, `text` | Option of a multiple-choice probe ("A", "B", ...). |
| `Probe` | `id`, `claim_id`, `question`, `choices`, `answer`, `rubric`, `difficulty`, `p_true`, `p_false`, `data` | Empty `choices` means a free-text answer. `answer` is the correct choice id, or a reference answer for free text. |
| `Observation` | `probe_id`, `claim_id`, `answer`, `score`, `feedback` | `score` is in `[0, 1]`. |
| `Turn` | `probe`, `observation` | One answered question. |
| `SessionState` | `claims`, `probes`, `belief`, `max_questions`, `graph`, `inputs`, `history`, `notes`, `data` | What policies see. |
| `ClaimResult` | `claim`, `probability`, `status`, `questions`, `explanation` | |
| `Verdict` | `status`, `probability`, `claims`, `notes` | `probability` = lowest claim probability. |

Statuses: `supported`, `refuted`, `uncertain`.

## Interfaces (`interfaces.py`)

| Class | Method | Contract |
| --- | --- | --- |
| `ClaimExtractor` | `extract(inputs) -> list[Claim]` | Raise `ValueError` with a user-readable message for bad input. May return `[]`; the engine then raises `ValueError("No claims found to verify.")`. |
| `KnowledgeSource` | `build(claims, inputs) -> KnowledgeGraph` | Record provenance on edges. |
| `ProbeGenerator` | `generate(claims, graph) -> list[Probe]` | Probe ids unique per session. Set `p_true > p_false`. |
| `Assessor` | `assess(probe, answer) -> Observation` | Never raises for a normal answer; scores in `[0, 1]`. |
| `Policy` | `reset(state)`, `choose(state) -> Probe or None`, `observe(state, probe, observation)` | `None` means stop. Must only return probes from `state.probes` that were not asked yet. |
| `Respondent` | `answer(probe) -> str` | Simulation only. |

`ChoiceAssessor` (default) scores 1.0 when the picked choice id equals `probe.answer`.

## Engine (`engine.py`)

`Session(pack, inputs, policy=None)`:
1. claims = `pack.claim_extractor().extract(inputs)`; empty → `ValueError`.
2. graph = `pack.knowledge_source().build(claims, inputs)`.
3. probes = `pack.probe_generator().generate(claims, graph)`.
4. belief = `pack.belief_model()`; `belief.start(claims)`.
5. policy = given policy or `pack.policy()`; `policy.reset(state)`; ask the first probe.

`session.answer(text)`:
- Raises `SessionError` if finished; `ValueError` for an invalid choice id or an empty free-text answer.
- Grades with the assessor, updates the belief, appends a `Turn`, calls `policy.observe`, then
  asks the next probe.

The session finishes (and builds `session.verdict`) when the policy returns `None` or
`max_questions` answers were given.

Verdict per claim: the belief's status and probability, plus an explanation
"`<passed> of <n> answers passed; <p>% likely true.`" or "Not tested.".
Overall: `refuted` if any claim is refuted; `supported` if all are supported; otherwise `uncertain`.

## Belief model (`belief.py`)

Per claim, log-odds starting at `logit(prior)`. For an observation with score `s` on a probe
with pass rates `p_true`, `p_false`:

```
if_true  = s * p_true  + (1 - s) * (1 - p_true)
if_false = s * p_false + (1 - s) * (1 - p_false)
log_odds += log(if_true / if_false)
```

So `s = 1` is a pass, `s = 0` a fail, `s = 0.5` is neutral. Probabilities are clamped to
`[0.01, 0.99]`. Status: `supported` if `p >= accept`, `refuted` if `p <= reject`, else `uncertain`.

`expected_gain(probe)`: expected drop in entropy (bits) of the probe's claim from asking it.

## Policies (`policies.py`)

- `open_candidates(state)`: unasked probes whose claim is still `uncertain`.
- `verdict_settled(state)`: true once any claim is refuted.
- `greedy`: stop if the verdict is settled or no candidates; otherwise take the claim with the
  lowest probability among candidates, and ask its probe with the highest `expected_gain`.
- `random`: random open candidate. Baseline only.
- `learned`: see [rl.md](rl.md).

## Domain pack (`pack.py`)

`DomainPack` attributes: `name`, `title`, `description`, `input_fields`, `max_questions`,
`show_feedback`, `prior`, `accept`, `reject`.
Required overrides: `claim_extractor()`, `knowledge_source()`, `probe_generator()`, `sample_case(rng)`.
Optional overrides: `assessor()`, `belief_model()`, `policy(name, rng)`, `policy_names()`, `respondent(truth, rng)`.
The constructor takes `llm` (an `LLM` or `None`).

`input_fields` items: `{"name", "label", "type": "text"|"textarea", "required", "placeholder"}`.
The UI renders them as the start form; their values arrive as `inputs[name]` (strings).

## Simulation (`simulation.py`)

- `StatisticalRespondent(truth, rng, spread=0.1)`: passes a probe with `p_true` (claim true) or
  `p_false` (claim false), shifted by a per-person offset in `[-spread, spread]`. Passing means
  answering `probe.answer`; failing means a wrong choice, or "I'm not sure." for free text.
- `run_episode(pack, policy, rng) -> (session, truth)`.
- `episode_reward(session, truth)`: `+1` correct, `-0.25` uncertain, `-1` wrong, minus `0.02` per question.
- `evaluate(pack, policy_name, episodes, seed)` → accuracy, uncertain, wrong, avg_questions, reward.
  The expected verdict is `supported` when every claim in `truth` is true, otherwise `refuted`.

## Knowledge graph (`graph.py`)

`KnowledgeGraph(nodes, edges, data)` with `add_node(id, label, **attrs)`,
`add_edge(source, relation, target, provenance)`, `neighbours(id, relation=None)`, `to_dict()`.
`data` holds pack-specific extras (e.g. the identity pack's preprocessed legacy graph).
