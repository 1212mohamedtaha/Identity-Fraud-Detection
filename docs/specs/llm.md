# Spec: LLM layer

Code: `verity/llm/`.

## Interface

```python
class LLM:
    provider: str
    model: str
    def complete(self, system: str, prompt: str, max_tokens: int = 4000) -> str: ...
```

That is the whole contract. Providers raise `LLMError` for any failure (network, HTTP
error, refusal, empty reply). Callers catch `LLMError` and fall back to offline logic.

## Providers

| `VERITY_LLM_PROVIDER` | Class | Settings |
| --- | --- | --- |
| `claude` | `ClaudeLLM` | `ANTHROPIC_API_KEY`; `VERITY_LLM_MODEL` (default `claude-opus-5-5`) |
| `openai` | `OpenAICompatibleLLM` | `VERITY_LLM_MODEL` (required), `VERITY_LLM_BASE_URL` (default `https://api.openai.com/v1`), `VERITY_LLM_API_KEY` |
| `none` | – | `get_llm()` returns `None`: offline mode |

If `VERITY_LLM_PROVIDER` is unset: `claude` when `ANTHROPIC_API_KEY` is set, else `none`.

`ClaudeLLM` details:
- Uses the official `anthropic` SDK, one `messages.create` call per completion.
- For `claude-opus-5-5`, `claude-opus-5`, `claude-fable-5-1` and `claude-sonnet-5-5` it enables
  server-side refusal fallbacks (`betas=["server-side-fallback-2026-07-01"]`, `fallbacks="default"`):
  if the model declines, the API retries on a fallback model instead of failing.
- A final `stop_reason == "refusal"` raises `LLMError`.

`OpenAICompatibleLLM` posts to `{base_url}/chat/completions` with `httpx`.

`FakeLLM(replies)` (tests): `replies` is a list (returned in order) or a function
`(system, prompt) -> str`. Calls are recorded in `.calls`.

## Prompts

- Files: `verity/packs/<pack>/prompts/<name>.v<N>.md`.
- Format: system part, a line containing only `---`, user part.
- Placeholders: `{{name}}`, filled by `load_prompt(path, **values)`.
- User-provided text goes inside tags (`<cv>…</cv>`) and the system part says to treat it as data.
- Prompts that need structured output describe the exact JSON shape and say "Reply with JSON only."

## Helpers

- `ask_json(llm, prompt_path, **values)`: render the prompt, call the model, parse the first
  JSON object or array in the reply (code fences tolerated). On invalid JSON it retries once
  with a reminder, then raises `LLMError`. Every call is logged (`verity.llm` logger) with
  prompt file name, model, attempt and duration.
- `extract_json(text)`, `load_prompt(path, **values)`, `split_prompt(text)`.

## Cost notes

The CV pack makes 2 calls when a session starts (claims, questions) and 1 per answer
(grading). A 10-question session is about 12 calls. Simulations and training never call
an LLM (`verity evaluate` and `verity train` build packs with `llm=None`).
