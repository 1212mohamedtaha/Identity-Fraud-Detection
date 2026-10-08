import json
from types import SimpleNamespace

import httpx
import pytest

from verity.llm import LLMError, ask_json, get_llm
from verity.llm.base import extract_json, load_prompt, split_prompt
from verity.llm.claude import ClaudeLLM
from verity.llm.fake import FakeLLM
from verity.llm.openai_compatible import OpenAICompatibleLLM


def test_no_key_means_offline(monkeypatch):
    assert get_llm() is None


def test_provider_selection(monkeypatch):
    monkeypatch.setenv("VERITY_LLM_PROVIDER", "openai")
    monkeypatch.setenv("VERITY_LLM_MODEL", "llama3")
    monkeypatch.setenv("VERITY_LLM_BASE_URL", "http://localhost:11434/v1")
    llm = get_llm()
    assert isinstance(llm, OpenAICompatibleLLM) and llm.model == "llama3"
    monkeypatch.setenv("VERITY_LLM_PROVIDER", "nope")
    with pytest.raises(ValueError):
        get_llm()


def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    with pytest.raises(LLMError):
        extract_json("no json here")


def test_prompt_placeholders_and_sections(tmp_path):
    path = tmp_path / "p.v1.md"
    path.write_text("system {{x}}\n---\nuser {{x}} {{y}}")
    system, user = split_prompt(load_prompt(path, x=1, y="two"))
    assert system == "system 1" and user == "user 1 two"


def test_ask_json_retries_once(tmp_path):
    path = tmp_path / "p.v1.md"
    path.write_text("s\n---\nu")
    llm = FakeLLM(["oops", '{"ok": true}'])
    assert ask_json(llm, path) == {"ok": True}
    assert len(llm.calls) == 2


def test_openai_compatible_provider_sends_chat_request():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    llm = OpenAICompatibleLLM("m", base_url="http://x/v1", api_key="k", transport=httpx.MockTransport(handler))
    assert llm.complete("sys", "hello") == "hi"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "sys"}
    assert seen["auth"] == "Bearer k"


def test_openai_compatible_errors_become_llm_errors():
    llm = OpenAICompatibleLLM("m", transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(LLMError):
        llm.complete("s", "u")


class FakeAnthropicClient:
    def __init__(self, stop_reason="end_turn"):
        self.requests = []
        reply = SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text="hello")])

        def create(**kwargs):
            self.requests.append(kwargs)
            return reply

        self.messages = SimpleNamespace(create=create)
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=create))


def test_claude_provider_uses_refusal_fallbacks_on_default_model():
    client = FakeAnthropicClient()
    assert ClaudeLLM(client=client).complete("s", "u") == "hello"
    request = client.requests[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["fallbacks"] == "default"


def test_claude_provider_reports_refusals():
    with pytest.raises(LLMError):
        ClaudeLLM(client=FakeAnthropicClient("refusal")).complete("s", "u")
