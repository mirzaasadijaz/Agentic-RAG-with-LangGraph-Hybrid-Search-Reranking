"""
Multi-provider LLM factory.

Two roles are exposed - get_grading_llm() for the high-frequency structured
decisions (routing, grading, query rewriting) and get_generation_llm() for
the final, user-facing answer - each independently configurable in
config.py. By default grading runs on Groq (fast + cheap) and generation
runs on Google Gemini (stronger output quality), but either role can point
at either provider, or the same one, purely through .env - nothing else in
the codebase needs to change.
"""
from . import config


def _build_chat_model(provider: str, model: str, temperature: float):
    provider = provider.lower()

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, temperature=temperature)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, temperature=temperature)

    raise ValueError(f"Unknown LLM provider {provider!r}. Use 'groq' or 'google'.")


def get_grading_llm(temperature: float = config.LLM_TEMPERATURE):
    """Fast/cheap model for routing, grading, and query rewriting."""
    return _build_chat_model(config.GRADING_LLM_PROVIDER, config.GRADING_LLM_MODEL, temperature)


def get_generation_llm(temperature: float = config.LLM_TEMPERATURE):
    """Higher-quality model for the final, user-facing answer."""
    return _build_chat_model(
        config.GENERATION_LLM_PROVIDER, config.GENERATION_LLM_MODEL, temperature
    )
