# 0001: A general claim-verification core with domain packs

**Status:** accepted

## Context
The project started as identity-fraud detection: a GNN + hierarchical RL model asking about
places near an applicant's home and work. The same loop (claim → knowledge → questions →
graded answers → verdict) fits many other problems: checking CV skills, fan knowledge,
language level, review authenticity.

## Decision
Split the code into a domain-agnostic **core** (types, engine, belief model, policies,
simulation, RL, API, UI) and **domain packs** that supply claim extraction, knowledge,
question generation and grading. Identity becomes one pack (keeping the original model as
its `legacy` policy); the CV skills check is the second pack and the reference example.

## Consequences
- New use cases need a pack, not engine changes.
- The core may not contain domain words (enforced by review; see AGENTS.md).
- The original model still runs, so new policies can be compared with it.
