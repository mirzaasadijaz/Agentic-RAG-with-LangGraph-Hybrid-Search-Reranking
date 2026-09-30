"""
Wires the nodes in src/nodes.py into the agentic control flow:

                              START
                                |
                        route_question
                     /          |          \\
             chitchat      web_search      retrieve
                |               |              |
               END              |        grade_documents
                                |         /          \\
                                |  transform_query    generate
                                |        |              |    \\
                                |     retrieve     (grounded &  (needs work)
                                |   (loop, capped)   on-topic)      |
                                |        :              |           |
                                +--------:------------ END    transform_query
                                         :                          |
                                         '--(also loops back to retrieve)

Greetings and small talk take the chitchat route straight to END, skipping
retrieval and grading entirely.

Every loop (no relevant docs -> rewrite -> retrieve again; ungrounded/
off-topic generation -> rewrite -> retrieve again) is bounded by
`retry_count` vs. `MAX_RETRIES` (see src/nodes.py), so the graph always
terminates.
"""
from langgraph.graph import END, START, StateGraph

from . import nodes
from .state import GraphState


def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", nodes.retrieve)
    workflow.add_node("grade_documents", nodes.grade_documents)
    workflow.add_node("generate", nodes.generate)
    workflow.add_node("transform_query", nodes.transform_query)
    workflow.add_node("web_search", nodes.web_search)
    workflow.add_node("chitchat", nodes.chitchat)

    workflow.add_conditional_edges(
        START,
        nodes.route_question,
        {
            "web_search": "web_search",
            "vectorstore": "retrieve",
            "chitchat": "chitchat",
        },
    )
    workflow.add_edge("chitchat", END)
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        nodes.decide_to_generate,
        {
            "transform_query": "transform_query",
            "web_search": "web_search",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "retrieve")
    workflow.add_edge("web_search", "generate")
    workflow.add_conditional_edges(
        "generate",
        nodes.grade_generation_v_documents_and_question,
        {
            "useful": END,
            "not supported": "transform_query",
            "not useful": "transform_query",
        },
    )

    return workflow.compile()