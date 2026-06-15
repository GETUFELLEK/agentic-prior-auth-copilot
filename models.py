from typing import List, Literal, Optional
from pydantic import BaseModel, Field

class Grade(BaseModel):
    sufficient: bool = Field(description="context is sufficient AND relevant to decide")
    reason: str

class PADecision(BaseModel):
    decision: Literal["APPROVE", "DENY", "NEEDS_INFO"]
    cited_clauses: List[str]
    rationale: str
    missing_info: Optional[str] = None

class Critique(BaseModel):
    grounded: bool = Field(description="every claim is supported by cited context, no hallucination")
    reason: str