# Guide: choose or add an LLM provider

Verity works without any LLM. With one, the CV pack reads the CV itself, writes questions
about your projects and grades answers like an interviewer.

Settings are environment variables (copy `.env.example` for reference). On Windows, use
`set NAME=value` instead of `export NAME=value`.

## Claude (default)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
verity serve
```

- The default model is `claude-opus-5-5`. Use another one with
  `export VERITY_LLM_MODEL=claude-sonnet-5-5` (cheaper) or `claude-haiku-5-5` (cheapest).
- If the model declines a request, Verity asks the API to retry on a fallback model
  (server-side refusal fallback), and otherwise falls back to offline logic.

Check what is active: `curl http://127.0.0.1:8000/api/health`.

## Any OpenAI-compatible API

Many providers and local servers speak the same "chat completions" API:

```bash
export VERITY_LLM_PROVIDER=openai
export VERITY_LLM_MODEL=<model name>
export VERITY_LLM_BASE_URL=<base url>      # default https://api.openai.com/v1
export VERITY_LLM_API_KEY=<key>            # if the provider needs one
```

| Provider | `VERITY_LLM_BASE_URL` |
| --- | --- |
| OpenAI | `https://api.openai.com/v1` |
| Ollama (local, free) | `http://localhost:11434/v1` |
| LM Studio (local) | `http://localhost:1234/v1` |
| vLLM | `http://<host>:8000/v1` |
| Groq | `https://api.groq.com/openai/v1` |
| OpenRouter | `https://openrouter.ai/api/v1` |

Small local models may produce weaker questions and grading; the JSON parser retries once
and falls back to offline logic if a reply is unusable.

## Offline

```bash
export VERITY_LLM_PROVIDER=none
```

## Add a new provider

1. Create `verity/llm/myprovider.py`:

   ```python
   from .base import LLM, LLMError

   class MyProviderLLM(LLM):
       provider = "myprovider"

       def __init__(self, model):
           self.model = model

       def complete(self, system, prompt, max_tokens=4000):
           try:
               ...  # call the API, return the reply text
           except Exception as exc:
               raise LLMError(str(exc)) from exc
   ```

2. Add a branch in `get_llm()` in `verity/llm/__init__.py`.
3. Add a test that mocks the network (see `tests/test_llm.py`), then document it here.

## Writing prompts

- One file per prompt: `verity/packs/<pack>/prompts/<name>.v1.md`.
- System part, a line with only `---`, then the user part. Placeholders look like `{{cv}}`.
- Put user text inside tags (`<cv>…</cv>`) and say it is data, not instructions.
- Describe the exact JSON reply shape and say "Reply with JSON only."
- Changing what a prompt asks for? Create `.v2.md` and switch the code to it, so results
  stay comparable.
