"""
Structured-output schemas for every LLM decision point in the graph. Using
`llm.with_structured_output(Schema)` instead of parsing free text makes each
routing/grading decision a reliable, typed result rather than a string
we hope looks like "yes" or "no".
"""
from typing import Literal

from pydantic import BaseModel, Field


class RouteQuery(BaseModel):
    """Route a user message to the best destination."""

    datasource: Literal["vectorstore", "web_search", "chitchat"] = Field(
        description=(
            "'chitchat' for greetings, thanks, or small talk; "
            "'vectorstore' for questions about the user's uploaded documents; "
            "'web_search' for current events or topics unrelated to those documents."
        )
    )


class GradeDocuments(BaseModel):
    """Binary relevance score for a single retrieved document."""

    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the document is relevant to the question, else 'no'."
    )


class GradeHallucinations(BaseModel):
    """Binary score for whether a generation is grounded in the retrieved facts."""

    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the answer is grounded in the supplied facts, else 'no'."
    )


class GradeAnswer(BaseModel):
    """Binary score for whether a generation actually addresses the question."""

    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the answer addresses the question, else 'no'."
    )