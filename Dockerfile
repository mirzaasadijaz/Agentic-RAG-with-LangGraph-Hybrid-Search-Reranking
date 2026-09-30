# Agentic RAG (LangGraph) - hybrid search + reranking, served through a
# Streamlit UI (app.py) with the CLI scripts available as one-off commands.
FROM python:3.12-slim

# - Unbuffered output, so the agent's step-by-step prints show up immediately.
# - No .pyc files and no pip cache baked into the image.
# - Model caches live under /app/.cache so a single volume persists them (see
#   docker-compose.yml): HF_HOME for the HuggingFace embedding model,
#   RERANKER_CACHE_DIR for the FlashRank reranker.
# - Streamlit listens on 0.0.0.0:8501 in headless mode, with telemetry off and
#   the file watcher disabled (no hot reload in a container, and the watcher
#   logs noisy torch.classes errors when torch is installed).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    RERANKER_CACHE_DIR=/app/.cache/flashrank \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Run as an unprivileged user (uid 1000). /app/chroma_db and /app/.cache are
# created here, owned by that user, so the named volumes mounted over them in
# docker-compose.yml inherit the right ownership on first use.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/chroma_db /app/.cache \
    && chown appuser:appuser /app /app/chroma_db /app/.cache

# CPU-only PyTorch first. The default Linux wheel on PyPI bundles CUDA
# libraries (several GB) that a CPU-only container never uses, and
# sentence-transformers would otherwise pull that build in. Once torch is
# installed, requirements.txt treats it as already satisfied.
# To use PyPI's default torch build instead (e.g. for GPU), build with:
#   docker build --build-arg TORCH_INDEX_URL=https://pypi.org/simple .
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# Left unpinned on purpose, matching the floor pins in requirements.txt; pin
# it yourself if you need bit-for-bit reproducible builds.
# hadolint ignore=DL3013
RUN pip install torch --index-url ${TORCH_INDEX_URL}

# Dependencies before source, so code edits don't invalidate this layer.
# requirements.txt must include streamlit (used by app.py).
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=appuser:appuser . .
USER 1000:1000

# The Streamlit UI listens here.
EXPOSE 8502

# `python` is the entrypoint, so any script in the project still runs by name:
#   ingest.py  |  compare_retrieval.py "question"
# With no arguments the container starts the Streamlit UI (app.py).
ENTRYPOINT ["python"]
CMD ["-m", "streamlit", "run", "app.py"]