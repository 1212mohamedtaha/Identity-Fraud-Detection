"""Claude (Anthropic) provider. Needs ANTHROPIC_API_KEY."""
from .base import LLM, LLMError

DEFAULT_MODEL = "claude-opus-5-5"

# Models that accept server-side refusal fallbacks ("fallbacks": "default").
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}


class ClaudeLLM(LLM):
    provider = "claude"

    def __init__(self, model=None, client=None):
        import anthropic

        self.model = model or DEFAULT_MODEL
        self.client = client or anthropic.Anthropic()

    def complete(self, system, prompt, max_tokens=4000):
        request = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.model in _FALLBACK_MODELS:
            # If the model declines, the API retries on a fallback model instead of failing.
            response = self.client.beta.messages.create(
                **request, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        else:
            response = self.client.messages.create(**request)

        if response.stop_reason == "refusal":
            raise LLMError("The model declined this request.")
        text = "".join(block.text for block in response.content if block.type == "text")
        if not text:
            raise LLMError("The model returned no text.")
        return text
