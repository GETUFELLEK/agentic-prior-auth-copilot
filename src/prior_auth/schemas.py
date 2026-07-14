
"""Pydantic schemas shared across the agent, API, and evaluations."""

from typing import Literal

from pydantic import BaseModel, Field


DecisionType = Literal["APPROVE", "DENY", "NEEDS_INFO"]
ReviewMode = Literal["interactive", "skip"]


class Grade(BaseModel):
    """Assessment of whether retrieved policy context is usable."""

    sufficient: bool = Field(
        description=(
            "Whether the retrieved policy context is both relevant "
            "and sufficient to make a coverage decision."
        )
    )
    reason: str = Field(
        min_length=1,
        description="Explanation supporting the sufficiency assessment.",
    )


class PADecision(BaseModel):
    """Structured prior-authorization decision."""

    decision: DecisionType
    cited_clauses: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)
    missing_info: str | None = None


class Critique(BaseModel):
    """Grounding assessment of a proposed decision."""

    grounded: bool = Field(
        description=(
            "Whether every material claim is supported by the "
            "retrieved policy context."
        )
    )
    reason: str = Field(
        min_length=1,
        description="Explanation of the grounding assessment.",
    )


class DecisionRequest(BaseModel):
    """API request for a prior-authorization decision."""

    question: str = Field(
        min_length=5,
        max_length=5000,
        description="Clinical coverage or prior-authorization question.",
        examples=[
            "Is an MRI covered for a patient with a cardiac pacemaker?"
        ],
    )


class DecisionResponse(BaseModel):
    """API response containing the structured agent decision."""

    request_id: str
    decision: DecisionType
    rationale: str
    cited_clauses: list[str]
    missing_info: str | None = None

    retrieval_attempts: int = Field(ge=0)
    critique_attempts: int = Field(ge=0)
    human_review_status: str


class HealthResponse(BaseModel):
    """Operational health-check response."""

    status: Literal["healthy"]
    service: str
    version: str
    environment: str
