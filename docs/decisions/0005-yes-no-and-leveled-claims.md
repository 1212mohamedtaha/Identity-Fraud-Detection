# 0005: One model for yes/no and leveled claims

**Status:** accepted

## Context
Identity claims are yes/no ("works at X"). Skill claims have levels ("knows SQL at senior
level"): passing easy questions should not support a senior claim. We did not want two
belief models or two code paths in the core.

## Decision
- Every claim has an ordered scale `levels` and a `claimed_level`; it holds when the real
  level is at or above the claimed one. Yes/no is the two-level scale `("false", "true")`.
- The belief model keeps a probability per level; probes give a pass chance per level
  (`pass_rates`). Packs can set them directly, use `irt_pass_rates` (item response theory),
  or set only `p_true` / `p_false`, which reproduces the old yes/no model exactly.
- Simulated truth is a real level per claim (bools still accepted for yes/no packs).

## Consequences
- The identity pack and the legacy model work unchanged; their evaluation numbers are identical.
- The CV pack judges the claimed level, which makes it a much harder (and more honest)
  problem: with 3–4 questions per skill, levels are hard to tell apart. More questions per
  skill and better policies are the next steps.
