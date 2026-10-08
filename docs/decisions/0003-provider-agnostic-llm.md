# 0003: Provider-agnostic LLM layer, Claude by default

**Status:** accepted

## Context
LLMs make free-text use cases possible (reading CVs, writing questions, grading answers).
We want the best model by default, the freedom to use any provider (including local
models), and an app that still works with no key at all.

## Decision
- One tiny interface: `LLM.complete(system, prompt) -> str`.
- Providers: Claude (default, `claude-opus-5-5`), any OpenAI-compatible API (OpenAI, Ollama,
  vLLM, Groq, …), and a fake for tests. Selected with environment variables.
- Prompts are versioned `.md` files; structured replies are parsed as JSON with one retry.
- Every LLM feature has an offline fallback; tests never call a real model.

## Consequences
- Adding a provider is one small class.
- We don't use provider-specific features beyond what `complete()` needs (kept simple).
- Offline mode is weaker but always available, which also keeps tests and CI free.
