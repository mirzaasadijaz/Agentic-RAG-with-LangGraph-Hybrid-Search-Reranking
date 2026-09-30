"""
Tests for src/llm.py's provider switching. Construction only (never
.invoke()), so no real API calls happen - but ChatGroq / ChatGoogleGenerativeAI
both validate their API key at construction time, so tests set dummy keys
via monkeypatch rather than needing real ones.
"""
import pytest

from src import config
from src.llm import _build_chat_model, get_generation_llm, get_grading_llm


@pytest.fixture(autouse=True)
def dummy_api_keys(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "dummy_test_key")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy_test_key")


def test_build_chat_model_groq():
    llm = _build_chat_model("groq", "llama-3.3-70b-versatile", 0.0)
    assert type(llm).__name__ == "ChatGroq"


def test_build_chat_model_google():
    llm = _build_chat_model("google", "gemini-3.8-flash", 0.0)
    assert type(llm).__name__ == "ChatGoogleGenerativeAI"


def test_build_chat_model_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        _build_chat_model("openai", "gpt-4", 0.0)


def test_get_grading_llm_uses_configured_provider():
    llm = get_grading_llm()
    assert type(llm).__name__ == {"groq": "ChatGroq", "google": "ChatGoogleGenerativeAI"}[
        config.GRADING_LLM_PROVIDER
    ]


def test_get_generation_llm_uses_configured_provider():
    llm = get_generation_llm()
    assert type(llm).__name__ == {"groq": "ChatGroq", "google": "ChatGoogleGenerativeAI"}[
        config.GENERATION_LLM_PROVIDER
    ]


def test_structured_output_chain_builds_for_both_providers():
    """Exercises the exact pattern chains.py uses for every grader/router."""
    from pydantic import BaseModel

    class _Schema(BaseModel):
        binary_score: str

    for provider, model in [("groq", "llama-3.3-70b-versatile"), ("google", "gemini-3.8-flash")]:
        llm = _build_chat_model(provider, model, 0.0)
        chain = llm.with_structured_output(_Schema)
        assert chain is not None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
