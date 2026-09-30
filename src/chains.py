"""
Builds every LCEL chain the graph nodes call. Each builder is a plain
function (not a module-level singleton) so nodes always get a fresh chain
bound to the currently configured model, and so tests can monkeypatch them
independently.
"""
from langchain_core.output_parsers import StrOutputParser

from . import prompts
from .llm import get_generation_llm, get_grading_llm
from .schemas import GradeAnswer, GradeDocuments, GradeHallucinations, RouteQuery

# Router, graders, and the rewriter are all small structured-output calls
# that run often (once per retrieved doc, for example) - these use the
# "grading" role (Groq by default). Final answer generation uses the
# "generation" role (Google Gemini by default).


def _structured(llm, schema):
    """
    Prefer JSON-schema mode: the model returns schema-constrained JSON, so a
    greeting can't trigger Groq's "Tool choice is required, but model did not
    call a tool" error. Falls back to tool calling if the installed client
    doesn't support the argument (fix: pip install -U langchain-groq).
    """
    try:
        return llm.with_structured_output(schema, method="json_schema")
    except (TypeError, ValueError, NotImplementedError):
        return llm.with_structured_output(schema)


def build_router_chain():
    return prompts.router_prompt | _structured(get_grading_llm(), RouteQuery)


def build_retrieval_grader():
    return prompts.grade_doc_prompt | _structured(get_grading_llm(), GradeDocuments)


def build_rag_chain():
    return prompts.rag_prompt | get_generation_llm() | StrOutputParser()


def build_chitchat_chain():
    """Answers greetings and small talk directly, with no retrieval."""
    return prompts.chitchat_prompt | get_generation_llm() | StrOutputParser()


def build_hallucination_grader():
    return prompts.hallucination_prompt | _structured(get_grading_llm(), GradeHallucinations)


def build_answer_grader():
    return prompts.answer_grade_prompt | _structured(get_grading_llm(), GradeAnswer)


def build_rewriter():
    return prompts.rewrite_prompt | get_grading_llm() | StrOutputParser()