
from typing import Literal

from pydantic import BaseModel, Field


class DecisionRequest(BaseModel):
    question: str = Field(
        min_length=5,
        max_length=5000,
        description="Prior-authorization coverage question.",
        examples=[
            "Is an MRI covered for a patient with a cardiac pacemaker?"
        ],
    )


class DecisionResponse(BaseModel):
    request_id: str
    decision: Literal["APPROVE", "DENY", "NEEDS_INFO"]
    rationale: str
    cited_clauses: list[str]
    missing_info: str | None = None
    retrieval_attempts: int
    critique_attempts: int


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    service: str
