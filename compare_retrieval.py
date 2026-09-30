"""
Shows exactly what reranking buys you: prints hybrid (dense+BM25) results
BEFORE reranking, then the same candidate set AFTER cross-encoder reranking,
side by side. Run `python ingest.py` first.

Usage:
    python compare_retrieval.py "your question here"
"""
import sys

from src import config
from src.retrieval import get_hybrid_retriever, get_reranker


def main():
    query = " ".join(sys.argv[1:]) or "What happens if I go over my storage quota?"
    print(f"Query: {query!r}\n")

    hybrid = get_hybrid_retriever()
    hybrid_docs = hybrid.invoke(query)
    print(f"--- Hybrid search (dense + BM25 fusion), {len(hybrid_docs)} candidates ---")
    for i, d in enumerate(hybrid_docs):
        print(f"{i + 1}. {d.page_content[:90].strip()}...")

    reranker = get_reranker()
    reranked_docs = reranker.compress_documents(hybrid_docs, query)
    print(f"\n--- After cross-encoder reranking, top {len(reranked_docs)} ---")
    for i, d in enumerate(reranked_docs):
        score = d.metadata.get("relevance_score")
        score_str = f" (score={score:.3f})" if score is not None else ""
        print(f"{i + 1}.{score_str} {d.page_content[:90].strip()}...")

    print(
        f"\nNote: hybrid search alone returns up to "
        f"{config.DENSE_K + config.SPARSE_K} candidates ranked by fused "
        f"dense/BM25 scores; reranking rescores each (query, doc) pair "
        f"jointly and keeps only the top {config.RERANK_TOP_N}, which is what "
        f"the agent actually reasons over."
    )


if __name__ == "__main__":
    main()
