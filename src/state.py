"""
The shared state that flows through every node of the graph. Each node
receives the current state and returns a partial dict of the keys it wants
to update; LangGraph merges that into the running state.
"""
from typing import List, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """
    Attributes:
        question: the current question driving retrieval (may be rewritten
            by transform_query as the agent tries to improve recall).
        original_question: the question exactly as the user asked it.
        generation: the LLM's answer, once produced.
        documents: the current working set of retrieved (and reranked) docs.
        web_search: "Yes" if document grading found nothing relevant and a
            web search fallback is needed, else "No".
        retry_count: how many times we've looped back through transform_query.
            Bounds every cycle in the graph so it always terminates.
    """

    question: str
    original_question: str
    generation: str
    documents: List[Document]
    web_search: str
    retry_count: int
