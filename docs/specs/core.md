# Spec: core

Code: `verity/core/`. The core must stay domain-agnostic (no identity, CV or other domain words).

## Data types (`types.py`)

| Type | Fields | Notes |
| --- | --- | --- |
| `Claim` | `id`, `text`, `kind`, `data`, `levels`, `claimed_level` | `id` is unique within a session. `data` is free space for the pack. See "Yes/no and leveled claims" below. |
| `Choice` | `id`, `text` | Option of a multiple-choice probe ("A", "B", ...). |
| `Probe` | `id`, `claim_id`, `question`, `choices`, `answer`, `rubric`, `difficulty`, `p_true`, `p_false`, `pass_rates`, `data` | Empty `choices` means a free-text answer. `answer` is the correct choice id, or a reference answer for free text. `pass_rates[level]` = chance of passing at each level of the claim. |
| `Observation` | `probe_id`, `claim_id`, `answer`, `score`, `feedback` | `score` is in `[0, 1]`. |
| `Turn` | `probe`, `observation` | One answered question. |
| `SessionState` | `claims`, `probes`, `belief`, `max_questions`, `graph`, `inputs`, `history`, `notes`, `data` | What policies see. |
| `ClaimResult` | `claim`, `probability`, `status`, `questions`, `explanation`, `level` | `level` = most likely real level (index). |
| `Verdict` | `status`, `probability`, `claims`, `notes` | `probability` = lowest claim probability. |

Statuses: `supported`, `refuted`, `uncertain`.

### Yes/no and leveled claims

Every claim lives on an ordered scale `levels` (lowest first) and asserts a minimum level
`claimed_level` (an index; default: the top level). The claim **holds** when the person's
real level is `>= claimed_level` (`claim.holds_at(level)`).

| Kind | `levels` | `claimed_level` |
| --- | --- | --- |
| Yes/no (default) | `("false", "true")` | 1 ("true") |
| Skill (CV pack) | `("none", "beginner", "junior", "mid", "senior")` | what the CV says |

Probes describe their strength per level with `pass_rates`. A pack may instead set only
`p_true` / `p_false`; the engine then fills `pass_rates` with `p_false` below the claimed level
and `p_true` at or above it (`belief.default_pass_rates`). For yes/no claims that is
`[p_false, p_true]`, identical to a classic two-hypothesis test.
`belief.irt_pass_rates(n_levels, difficulty, discrimination=1.7, guess=0.05, slip=0.05)`
builds level-based rates from item response theory: `guess + (1 - guess - slip) * sigmoid(discrimination * (level - difficulty))`.

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
3. probes = `pack.probe_generator().generate(claims, graph)`; probes without `pass_rates` get
   `default_pass_rates(claim, probe)`.
4. belief = `pack.belief_model()`; `belief.start(claims)`.
5. policy = given policy or `pack.policy()`; `policy.reset(state)`; ask the first probe.

`session.answer(text)`:
- Raises `SessionError` if finished; `ValueError` for an invalid choice id or an empty free-text answer.
- Grades with the assessor, updates the belief, appends a `Turn`, calls `policy.observe`, then
  asks the next probe.

The session finishes (and builds `session.verdict`) when the policy returns `None` or
`max_questions` answers were given.

Verdict per claim: the belief's status, probability and most likely level, plus an explanation
"`<passed> of <n> answers passed; <p>% likely true.`" (leveled claims add
"` Most likely level: <level> (claimed: <level>).`"), or "Not tested.".
Overall: `refuted` if any claim is refuted; `supported` if all are supported; otherwise `uncertain`.

## Belief model (`belief.py`)

Per claim, a probability for every level. Start (`prior_for`): `prior` (default 0.5) spread
evenly over the levels where the claim holds, `1 - prior` over the levels below it
(yes/no: `[1 - prior, prior]`; a claim at level 0 starts uniform).

After an observation with score `s` on a probe with pass rates `r` (rates clamped to
`[0.01, 0.99]`), each level's weight is multiplied by the likelihood of `s` at that level:

- multiple-choice probe: `r[level] ** s * (1 - r[level]) ** (1 - s)` (= `r` for a pass, `1 - r` for a fail);
- free-text probe: `exp(-(s - r[level])² / (2 · score_noise²))`: `r[level]` is the expected score
  at that level, `score_noise` (default 0.2) the typical spread of real scores around it.

Then weights are renormalised, given a floor of 0.001 and renormalised again so no level is
ever ruled out by one answer. See decision 0006.

- `probability(claim_id)`: total weight on levels where the claim holds.
- `level(claim_id)`: most likely level.
- `status`: `supported` if `p >= accept`, `refuted` if `p <= reject`, else `uncertain`.
- `expected_gain(probe)`: expected drop in entropy (bits) of "the claim holds" from asking the probe,
  averaged over the possible answers (pass/fail, or free-text scores 0, 0.1, …, 1).

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
Optional overrides: `assessor()`, `belief_model()`, `policy(name, rng)`, `policy_names()`,
`respondent(truth, rng, case)`, `make_case(rng, index)`, `write_dataset_extras(cases, out, rng)`, `mock_llm(seed)`.
The constructor takes `llm` (an `LLM` or `None`).

`input_fields` items: `{"name", "label", "type": "text"|"textarea", "required", "placeholder"}`.
The UI renders them as the start form; their values arrive as `inputs[name]` (strings).

## Fitting (`fitting.py`)

`fit_item(observations, n_levels)`: for answers `(level, score)` to one question, the
`(difficulty, discrimination)` of `irt_pass_rates` that maximises
`Σ score·log r + (1 − score)·log(1 − r)`, found by grid search (difficulty −1..n_levels in 0.1
steps; discrimination 0.8, 1.2, 1.7, 2.4, 3.2).

## Simulation (`simulation.py`)

- `sample_case(rng)` returns `(inputs, truth)`; `truth[claim_id]` is the person's real level, or a
  bool for yes/no use cases. `true_levels(truth, claims)` converts: `True` → the claimed level,
  `False` → one level below it.
- `StatisticalRespondent(truth_levels, rng, spread=0.1)`: passes a probe with
  `probe.pass_rates[real level]`, shifted by a per-person offset in `[-spread, spread]`. Passing means
  answering `probe.answer`; failing means a wrong choice, or "I'm not sure." for free text.
- `run_episode(pack, policy, rng, case=None) -> (session, truth levels)`: plays a dataset case,
  or a fresh `pack.make_case(rng, 0)`; the respondent is `pack.respondent(truth, rng, case)`.
- `episode_reward(session, truth)`: `+1` correct, `-0.25` uncertain, `-1` wrong, minus `0.02` per question.
- `evaluate(pack, policy_name, episodes, seed, cases=None)` → accuracy, uncertain, wrong,
  avg_questions, reward. With `cases`, plays each case once.
- Datasets: see [dataset.md](dataset.md).
  The expected verdict is `supported` when every claim holds at the person's real level, otherwise `refuted`.

## Knowledge graph (`graph.py`)

`KnowledgeGraph(nodes, edges, data)` with `add_node(id, label, **attrs)`,
`add_edge(source, relation, target, provenance)`, `neighbours(id, relation=None)`, `to_dict()`.
`data` holds pack-specific extras (e.g. the identity pack's preprocessed legacy graph).
