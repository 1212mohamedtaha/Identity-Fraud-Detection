"""Any provider that speaks the OpenAI-style chat completions API.

That covers OpenAI, Ollama (local), LM Studio, vLLM, Groq, OpenRouter and many more:
set VERITY_LLM_BASE_URL, VERITY_LLM_MODEL and (if needed) VERITY_LLM_API_KEY.
"""
import httpx

from .base import LLM, LLMError


class OpenAICompatibleLLM(LLM):
    provider = "openai"

    def __init__(self, model, base_url="https://api.openai.com/v1", api_key="", timeout=120, transport=None):
        if not model:
            raise ValueError("Set VERITY_LLM_MODEL to the model name of your provider.")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.http = httpx.Client(timeout=timeout, transport=transport)

    def complete(self, system, prompt, max_tokens=4000):
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        try:
            response = self.http.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc
