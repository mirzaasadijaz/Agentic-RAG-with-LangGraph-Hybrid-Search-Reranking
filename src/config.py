"""
Central configuration. Every tunable knob for the agent lives here so the
rest of the codebase never hardcodes a model name, weight, or path.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root (this file lives in src/, so the root is one level up).
_BASE_DIR = Path(__file__).resolve().parent.parent


def _path_from_env(name: str, default: str) -> str:
    """
    Read a path from the environment and return it as an absolute path.
    Relative values (from .env or the default) are resolved against the
    project root, so they no longer depend on the folder you launch from.
    """
    p = Path(os.getenv(name, default)).expanduser()
    if not p.is_absolute():
        p = _BASE_DIR / p
    return str(p.resolve())


GRADING_LLM_PROVIDER = os.getenv("GRADING_LLM_PROVIDER", "groq")
GRADING_LLM_MODEL = os.getenv("GRADING_LLM_MODEL", "openai/gpt-oss-20b")

GENERATION_LLM_PROVIDER = os.getenv("GENERATION_LLM_PROVIDER", "google")
GENERATION_LLM_MODEL = os.getenv("GENERATION_LLM_MODEL", "gemini-flash-latest")

LLM_TEMPERATURE = 0.0

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
HF_TOKEN = os.getenv("HF_TOKEN")

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2")
RERANKER_CACHE_DIR = _path_from_env("RERANKER_CACHE_DIR", ".cache/flashrank")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

DENSE_K = 6
SPARSE_K = 6
DENSE_WEIGHT = 0.5
SPARSE_WEIGHT = 0.5
RERANK_TOP_N = 4

WEB_SEARCH_K = 3

MAX_RETRIES = 2

CHROMA_DIR = _path_from_env("CHROMA_DIR", "chroma_db")
COLLECTION_NAME = "agentic_rag_docs"
DATA_DIR = _path_from_env("DATA_DIR", "data/sample_docs")

_PROVIDER_ENV_VAR = {"groq": "GROQ_API_KEY", "google": "GOOGLE_API_KEY"}


def check_required_api_keys():
    """
    Raise one clear, actionable error listing every missing key, instead of
    letting the first LLM call fail with a less friendly client error in the
    middle of a conversation. Called by app.py / ingest.py at startup.
    """
    missing = {
        _PROVIDER_ENV_VAR[p.lower()]
        for p in (GRADING_LLM_PROVIDER, GENERATION_LLM_PROVIDER)
        if p.lower() in _PROVIDER_ENV_VAR and not os.getenv(_PROVIDER_ENV_VAR[p.lower()])
    }
    if missing:
        raise RuntimeError(
            f"Missing required API key(s): {', '.join(sorted(missing))}. "
            "Copy .env.example to .env and fill them in."
        )