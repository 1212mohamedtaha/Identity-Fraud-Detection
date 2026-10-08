"""Pick the LLM provider from environment variables.

    VERITY_LLM_PROVIDER   claude | openai | none   (default: claude if ANTHROPIC_API_KEY is set, else none)
    VERITY_LLM_MODEL      model name (Claude default: claude-opus-5-5)
    VERITY_LLM_BASE_URL   for "openai": the API base URL, e.g. http://localhost:11434/v1 for Ollama
    VERITY_LLM_API_KEY    for "openai": the API key, if the provider needs one

With provider "none" everything still works: packs fall back to their offline logic.
See docs/specs/llm.md.
"""
import os

from .base import LLM, LLMError, ask_json, load_prompt

__all__ = ["LLM", "LLMError", "ask_json", "get_llm", "load_prompt"]


def get_llm():
    provider = os.environ.get("VERITY_LLM_PROVIDER", "").strip().lower()
    if not provider:
        provider = "claude" if os.environ.get("ANTHROPIC_API_KEY") else "none"
    model = os.environ.get("VERITY_LLM_MODEL") or None

    if provider == "none":
        return None
    if provider == "claude":
        from .claude import ClaudeLLM
        return ClaudeLLM(model=model)
    if provider == "openai":
        from .openai_compatible import OpenAICompatibleLLM
        return OpenAICompatibleLLM(
            model=model,
            base_url=os.environ.get("VERITY_LLM_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.environ.get("VERITY_LLM_API_KEY", ""),
        )
    raise ValueError(f"Unknown VERITY_LLM_PROVIDER {provider!r}; use claude, openai or none.")
