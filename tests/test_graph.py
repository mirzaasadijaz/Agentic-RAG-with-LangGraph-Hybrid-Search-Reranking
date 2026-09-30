"""
Drives the real compiled graph with every LLM/retriever call mocked out, so
this suite needs no API key and runs in well under a second. It verifies
the parts that are easy to get subtly wrong in a graph like this: routing,
the two retry loops, and - most importantly - that retry_count actually
bounds both loops so the graph is guaranteed to terminate.
"""
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.graph import build_graph

RELEVANT_DOC = Document(page_content="Nimbus Free tier is 5GB.")
MAX_STEPS = 25  # generous ceiling; a hung/looping graph would blow past this


def _chain(return_value):
    return MagicMock(invoke=MagicMock(return_value=return_value))


class _Score:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _run_graph(router_dest, doc_relevant, hallucination_ok, answer_ok):
    """Runs the real compiled graph end-to-end with every LLM call mocked."""
    with patch("src.nodes.get_retriever") as mock_retriever, \
         patch("src.chains.build_router_chain") as mock_router, \
         patch("src.chains.build_retrieval_grader") as mock_grader, \
         patch("src.chains.build_rag_chain") as mock_rag, \
         patch("src.chains.build_hallucination_grader") as mock_halluc, \
         patch("src.chains.build_answer_grader") as mock_answer, \
         patch("src.chains.build_rewriter") as mock_rewrite, \
         patch("src.nodes.get_web_search_tool") as mock_web:

        mock_retriever.return_value = MagicMock(invoke=MagicMock(return_value=[RELEVANT_DOC]))
        mock_router.return_value = _chain(_Score(datasource=router_dest))
        mock_grader.return_value = _chain(_Score(binary_score="yes" if doc_relevant else "no"))
        mock_rag.return_value = _chain("This is a mocked answer.")
        mock_halluc.return_value = _chain(_Score(binary_score="yes" if hallucination_ok else "no"))
        mock_answer.return_value = _chain(_Score(binary_score="yes" if answer_ok else "no"))
        mock_rewrite.return_value = _chain("rewritten question")
        mock_web.return_value = MagicMock(
            invoke=MagicMock(return_value={"results": [{"content": "web fact"}]})
        )

        graph = build_graph()
        initial_state = {
            "question": "What is the free tier storage limit?",
            "original_question": "What is the free tier storage limit?",
            "documents": [],
            "generation": "",
            "web_search": "No",
            "retry_count": 0,
        }

        steps, final_state = 0, initial_state
        for state in graph.stream(initial_state, stream_mode="values"):
            steps += 1
            assert steps <= MAX_STEPS, "graph did not terminate - a loop is unbounded"
            final_state = state
        return final_state, steps


def test_graph_compiles():
    assert build_graph() is not None


def test_happy_path_terminates_with_answer():
    final_state, steps = _run_graph(
        router_dest="vectorstore", doc_relevant=True, hallucination_ok=True, answer_ok=True
    )
    assert final_state["generation"]
    assert final_state["retry_count"] == 0  # no loop needed
    assert steps < 10


def test_no_relevant_docs_falls_back_to_web_search_after_max_retries():
    final_state, _ = _run_graph(
        router_dest="vectorstore", doc_relevant=False, hallucination_ok=True, answer_ok=True
    )
    assert final_state["generation"]
    assert final_state["retry_count"] >= 2
    assert any(d.metadata.get("source") == "web_search" for d in final_state["documents"])


def test_hallucinating_generation_is_bounded_and_still_terminates():
    final_state, _ = _run_graph(
        router_dest="vectorstore", doc_relevant=True, hallucination_ok=False, answer_ok=True
    )
    assert final_state["generation"]
    assert final_state["retry_count"] >= 2


def test_router_can_send_straight_to_web_search():
    final_state, steps = _run_graph(
        router_dest="web_search", doc_relevant=True, hallucination_ok=True, answer_ok=True
    )
    assert final_state["generation"]
    assert any(d.metadata.get("source") == "web_search" for d in final_state["documents"])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
