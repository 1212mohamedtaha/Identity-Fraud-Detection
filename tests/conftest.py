import pytest

from tests.toy_pack import ToyPack


@pytest.fixture
def toy():
    return ToyPack()


@pytest.fixture(autouse=True)
def no_llm_env(monkeypatch, tmp_path):
    """Tests never call a real LLM, and trained policies go to a temp folder."""
    for name in ("VERITY_LLM_PROVIDER", "VERITY_LLM_MODEL", "VERITY_LLM_BASE_URL",
                 "VERITY_LLM_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
