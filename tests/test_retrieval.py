"""
Tests for the hybrid retrieval layer. These exercise the real functions in
src/retrieval.py. Chunking and BM25 (sparse) retrieval are pure-local and
need no network access. The dense/embedding half is patched out with a
lightweight fake retriever so this suite runs offline and fast; the
end-to-end pipeline (real embeddings + real reranker model) is exercised by
running `python ingest.py && python compare_retrieval.py "..."` yourself,
since that legitimately needs to download two small ONNX models on first
run.
"""
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from src import config
from src.retrieval import (
    build_vectorstore,
    get_hybrid_retriever,
    get_reranker,
    get_sparse_retriever,
    load_and_split_documents,
)


def test_load_and_split_documents_produces_chunks():
    chunks = load_and_split_documents(config.DATA_DIR)
    assert len(chunks) > 0
    # every source file should have contributed at least one chunk
    sources = {c.metadata.get("source") for c in chunks}
    assert len(sources) == 5
    assert all(len(c.page_content) <= config.CHUNK_SIZE + 50 for c in chunks)  # splitter slack


def test_sparse_retriever_finds_exact_error_code():
    """BM25 should nail an exact keyword/error-code match."""
    chunks = load_and_split_documents(config.DATA_DIR)
    retriever = get_sparse_retriever(chunks)
    results = retriever.invoke("ERR_507_QUOTA_EXCEEDED")
    assert len(results) > 0
    assert any("ERR_507_QUOTA_EXCEEDED" in d.page_content for d in results)


class _FakeDenseRetriever(BaseRetriever):
    """Stands in for the Chroma+embeddings retriever so this test needs no network."""

    docs: list

    def _get_relevant_documents(self, query, *, run_manager=None):
        return self.docs


def test_hybrid_retriever_fuses_dense_and_sparse():
    """
    Exercises the real get_hybrid_retriever() code path, with only the
    dense half faked out, and checks the EnsembleRetriever fusion actually
    returns documents found by BOTH sub-retrievers, not just one.
    """
    chunks = load_and_split_documents(config.DATA_DIR)
    dense_only_doc = Document(page_content="A completely dense-only fake result.")
    fake_dense = _FakeDenseRetriever(docs=[dense_only_doc])

    with patch("src.retrieval.get_dense_retriever", return_value=fake_dense):
        hybrid = get_hybrid_retriever()
        results = hybrid.invoke("What is the storage quota error code?")

    assert len(results) > 0
    # the fake dense retriever's unique doc should show up in the fused results
    assert any(d.page_content == dense_only_doc.page_content for d in results)
    # BM25 should also have contributed at least one real doc from the corpus
    assert any(d.page_content != dense_only_doc.page_content for d in results)


class _FakeEmbeddings(Embeddings):
    """Offline stand-in for the HuggingFace model, so index tests need no download."""

    def embed_documents(self, texts):
        return [[float(len(t) % 7), float(len(t) % 5), 1.0] for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def test_rebuilding_the_index_does_not_duplicate_chunks(tmp_path):
    """Regression: re-running ingest used to append a second copy of every chunk."""
    chunks = load_and_split_documents(config.DATA_DIR)
    with patch("src.retrieval.get_embeddings", return_value=_FakeEmbeddings()):
        first = build_vectorstore(chunks, persist_directory=str(tmp_path))
        assert len(first.get()["ids"]) == len(chunks)

        second = build_vectorstore(chunks, persist_directory=str(tmp_path))
        assert len(second.get()["ids"]) == len(chunks)


def test_reranker_uses_configured_model_and_cache_dir_and_respects_top_n():
    """
    Uses a real flashrank.Ranker subclass with only the model download
    faked, so this exercises the exact FlashrankRerank(client=...) wiring
    that get_reranker() ships with.
    """
    import flashrank

    class _FakeRanker(flashrank.Ranker):
        def __init__(self, model_name, cache_dir, **kwargs):  # skips the real download
            self.seen = {"model_name": model_name, "cache_dir": cache_dir}

        def rerank(self, request):
            ranked = [
                {"id": p["id"], "text": p["text"], "meta": p["meta"], "score": float(len(p["text"]))}
                for p in request.passages
            ]
            return sorted(ranked, key=lambda r: r["score"], reverse=True)

    with patch("flashrank.Ranker", _FakeRanker):
        reranker = get_reranker()

    assert reranker.client.seen == {
        "model_name": config.RERANKER_MODEL,
        "cache_dir": config.RERANKER_CACHE_DIR,
    }

    docs = [Document(page_content="x" * n) for n in range(1, config.RERANK_TOP_N + 4)]
    result = reranker.compress_documents(docs, "any query")
    assert len(result) == config.RERANK_TOP_N
    scores = [d.metadata["relevance_score"] for d in result]
    assert scores == sorted(scores, reverse=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
