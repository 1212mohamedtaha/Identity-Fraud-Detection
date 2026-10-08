# Architecture

Verity answers one question: **is this person's claim true?** It does so the way a good
interviewer would: ask a question, listen, update its opinion, and pick the next question
that would change its mind the most. It stops when it is sure or runs out of questions.

## The loop

```
                 ┌──────────────── DomainPack (one per use case) ────────────────┐
inputs ──► ClaimExtractor ──► KnowledgeSource ──► ProbeGenerator                  │
           (claims)           (knowledge graph)   (candidate probes)              │
                 └──────────────────────────┬─────────────────────────────────────┘
                                            ▼
                              ┌──────── Session (engine) ────────┐
                              │  Policy.choose(state) ─► probe   │◄── person answers
                              │  Assessor.assess ─► observation  │
                              │  BeliefModel.update              │
                              │  repeat until Policy says stop   │
                              │  or max_questions is reached     │
                              └───────────────┬──────────────────┘
                                              ▼
                                Verdict: status + probability per claim
```

Every box is a small class with one job. The engine (`verity/core/engine.py`) is the
only place they meet, so each can be replaced on its own.

## Building blocks

| Block | Job | Default in core | Identity pack | CV pack |
| --- | --- | --- | --- | --- |
| `ClaimExtractor` | inputs → claims | – | attributes stored in the applicant's graph | LLM reads the CV (keywords offline) |
| `KnowledgeSource` | claims → knowledge graph | – | places near each attribute | skill → topic graph |
| `ProbeGenerator` | graph → candidate probes | – | multiple-choice questions | LLM interview questions (question bank offline) |
| `Assessor` | answer → score 0..1 | `ChoiceAssessor` (exact match) | default | LLM rubric grading (keywords offline) |
| `BeliefModel` | scores → probability per claim | Bayesian log-odds | default | default (stricter thresholds) |
| `Policy` | which probe next, or stop | `greedy`, `random`, `learned` | + `legacy` (original RL model) | default |
| `Respondent` | simulated person for training | `StatisticalRespondent` | default | default |

A `DomainPack` subclass picks the implementation of each block by overriding small
factory methods (`claim_extractor()`, `assessor()`, ...). See
[guides/adding-a-pack.md](guides/adding-a-pack.md).

## Key ideas

**Probes carry their own evidence strength.** Every probe has `p_true` (chance a person
for whom the claim is true passes it) and `p_false` (chance someone for whom it is false
passes, e.g. by guessing). The belief model turns each graded answer into evidence with
Bayes' rule. That is what makes the core domain-agnostic: it never needs to know *what*
a question is about, only how telling it is.

**One refuted claim refutes the set.** The overall verdict is `refuted` if any claim is
refuted, `supported` if all are supported, otherwise `uncertain`. Policies stop as soon as
the verdict cannot change any more.

**Policies are swappable and measurable.** `verity evaluate <pack>` runs every policy on
simulated people whose truth is known and prints accuracy, questions asked and reward.
The RL policy is trained on the same simulator (`verity train <pack>`).

**LLMs are optional and replaceable.** All LLM use goes through `verity.llm.LLM.complete()`.
Claude is the default; any OpenAI-compatible API works; with no provider the packs fall
back to offline logic. Prompts are versioned files next to the pack that uses them.

## Where the original project lives

The original GNN + hierarchical RL model, its checkpoint and its data are in
`verity/packs/identity/legacy/` and `verity/packs/identity/data/`. They run unchanged as
the identity pack's `legacy` policy, so its behaviour can be compared with the new policies.

## Further reading

- Specs: [core](specs/core.md), [LLM](specs/llm.md), [RL](specs/rl.md), [API and UI](specs/api.md),
  [identity pack](specs/pack-identity.md), [CV pack](specs/pack-cv.md)
- Guides: [training](guides/training.md), [adding a pack](guides/adding-a-pack.md),
  [LLM providers](guides/llm-providers.md)
- Decisions: [docs/decisions/](decisions/)
