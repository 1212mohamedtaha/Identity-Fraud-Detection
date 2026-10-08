# Decision records

Short notes on significant design choices: what we decided, why, and what we gave up.
Add a new numbered file when you make a choice someone might later question.

| # | Decision |
| --- | --- |
| [0001](0001-general-claim-verification-core.md) | A general claim-verification core with domain packs |
| [0002](0002-bayesian-belief-and-swappable-policies.md) | Bayesian belief per claim; policies (including RL) only choose questions |
| [0003](0003-provider-agnostic-llm.md) | Provider-agnostic LLM layer, Claude by default, offline fallback everywhere |
| [0004](0004-fastapi-and-plain-js-ui.md) | FastAPI for the API, plain HTML/JS for the UI |
