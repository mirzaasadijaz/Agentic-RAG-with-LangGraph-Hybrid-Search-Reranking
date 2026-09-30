"""
Every node and conditional-edge function in the graph.

Regular nodes (retrieve, grade_documents, generate, chitchat,
transform_query, web_search) take the state and return a partial state
update.

Conditional-edge functions (route_question, decide_to_generate,
grade_generation_v_documents_and_question) take the state and return a
plain string naming the next node - they cannot update state themselves,
which is why `retry_count` is only ever incremented inside transform_query.

Structured-output calls (router, graders) can fail when a small model
misbehaves (for example Groq "tool_use_failed"). Those calls go through
`safe_invoke`, which falls back to a safe default so a single bad response
never kills the whole run.
"""
from types import SimpleNamespace

from langchain_core.documents import Document

from . import chains, config
from .retrieval import get_retriever
from .state import GraphState
from .tools import get_web_search_tool


def format_docs(docs) -> str:
    return "\n\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))


def safe_invoke(chain, inputs: dict, default, label: str = "llm call"):
    """Run a chain; return `default` if the model call fails or misbehaves."""
    try:
        return chain.invoke(inputs)
    except Exception as e:
        print(f"  [warn] {label} failed, using default: {type(e).__name__}: {e}")
        return default


# Fallback results, shaped like the structured outputs the chains return.
_NOT_RELEVANT = SimpleNamespace(binary_score="no")
_GROUNDED = SimpleNamespace(binary_score="yes")   # accept rather than loop forever
_ANSWERS = SimpleNamespace(binary_score="yes")
_DEFAULT_ROUTE = SimpleNamespace(datasource="vectorstore")


# --------------------------------------------------------------------------- nodes


def retrieve(state: GraphState) -> dict:
    print("---RETRIEVE (hybrid search + rerank)---")
    documents = get_retriever().invoke(state["question"])
    return {"documents": documents}


def grade_documents(state: GraphState) -> dict:
    print("---GRADE DOCUMENT RELEVANCE---")
    grader = chains.build_retrieval_grader()
    filtered = []
    for doc in state["documents"]:
        result = safe_invoke(
            grader,
            {"question": state["question"], "document": doc.page_content},
            default=_NOT_RELEVANT,
            label="retrieval grader",
        )
        tag = "relevant" if result.binary_score == "yes" else "not relevant"
        print(f"  [{tag}] {doc.page_content[:60].strip()}...")
        if result.binary_score == "yes":
            filtered.append(doc)
    return {"documents": filtered, "web_search": "Yes" if not filtered else "No"}


def generate(state: GraphState) -> dict:
    print("---GENERATE---")
    rag_chain = chains.build_rag_chain()
    generation = rag_chain.invoke(
        {"context": format_docs(state["documents"]), "question": state["question"]}
    )
    return {"generation": generation}


def chitchat(state: GraphState) -> dict:
    print("---CHITCHAT---")
    generation = chains.build_chitchat_chain().invoke({"question": state["question"]})
    return {"generation": generation}


def transform_query(state: GraphState) -> dict:
    print("---TRANSFORM QUERY---")
    rewriter = chains.build_rewriter()
    better_question = safe_invoke(
        rewriter,
        {"question": state["question"]},
        default=state["question"],  # keep the original wording if rewriting fails
        label="query rewriter",
    )
    print(f"  rewritten query: {better_question!r}")
    return {"question": better_question, "retry_count": state.get("retry_count", 0) + 1}


def web_search(state: GraphState) -> dict:
    print("---WEB SEARCH (fallback)---")
    try:
        tool = get_web_search_tool()
        response = tool.invoke({"query": state["question"]})
    except Exception as e:
        print(f"  [warn] web search failed: {type(e).__name__}: {e}")
        response = {}
    results = response.get("results", []) if isinstance(response, dict) else []
    content = "\n\n".join(r.get("content", "") for r in results) or "No web results found."
    web_doc = Document(page_content=content, metadata={"source": "web_search"})
    return {"documents": state.get("documents", []) + [web_doc]}


# --------------------------------------------------------------- conditional edges


def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    route = safe_invoke(
        chains.build_router_chain(),
        {"question": state["question"]},
        default=_DEFAULT_ROUTE,
        label="router",
    )
    print(f"  routed to: {route.datasource}")
    return route.datasource


def decide_to_generate(state: GraphState) -> str:
    print("---DECIDE: GENERATE, REWRITE, OR WEB SEARCH?---")
    if state.get("web_search") == "Yes":
        if state.get("retry_count", 0) >= config.MAX_RETRIES:
            print("  no relevant local docs after max retries -> web_search")
            return "web_search"
        print("  no relevant docs yet -> transform_query")
        return "transform_query"
    print("  relevant docs found -> generate")
    return "generate"


def grade_generation_v_documents_and_question(state: GraphState) -> str:
    print("---GRADE GENERATION---")
    if state.get("retry_count", 0) >= config.MAX_RETRIES:
        print("  max retries reached -> accepting generation as-is")
        return "useful"

    hallucination = safe_invoke(
        chains.build_hallucination_grader(),
        {"documents": format_docs(state["documents"]), "generation": state["generation"]},
        default=_GROUNDED,
        label="hallucination grader",
    )
    if hallucination.binary_score == "no":
        print("  not grounded in documents -> transform_query")
        return "not supported"

    answer = safe_invoke(
        chains.build_answer_grader(),
        {"question": state["question"], "generation": state["generation"]},
        default=_ANSWERS,
        label="answer grader",
    )
    if answer.binary_score == "yes":
        print("  grounded and on-topic -> done")
        return "useful"
    print("  grounded but doesn't address the question -> transform_query")
    return "not useful"