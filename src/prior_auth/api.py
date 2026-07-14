
import os
import uuid

from dotenv import load_dotenv

load_dotenv()

# Prevent API requests from waiting for terminal input on DENY decisions.
os.environ.setdefault("EVAL_MODE", "1")

from fastapi import FastAPI, HTTPException

from agent import app as decision_graph
from agent import make_initial_state
from api_models import DecisionRequest, DecisionResponse, HealthResponse
from models import PADecision


api = FastAPI(
    title="Agentic Prior-Authorization Copilot",
    version="0.1.0",
    description=(
        "A policy-grounded prior-authorization decision-support API "
        "implemented with LangGraph."
    ),
)


@api.get("/health", response_model=HealthResponse, tags=["Operations"])
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service="agentic-prior-auth-copilot",
    )


@api.post(
    "/v1/decisions",
    response_model=DecisionResponse,
    tags=["Prior Authorization"],
)
def create_decision(request: DecisionRequest) -> DecisionResponse:
    request_id = str(uuid.uuid4())

    config = {
        "configurable": {
            "thread_id": request_id,
        }
    }

    try:
        result = decision_graph.invoke(
            make_initial_state(request.question),
            config,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="The prior-authorization workflow failed.",
        ) from exc

    raw_decision = result.get("decision")

    if raw_decision is None:
        raise HTTPException(
            status_code=500,
            detail="The workflow completed without producing a decision.",
        )

    if isinstance(raw_decision, dict):
        decision = PADecision.model_validate(raw_decision)
    else:
        decision = raw_decision

    return DecisionResponse(
        request_id=request_id,
        decision=decision.decision,
        rationale=decision.rationale,
        cited_clauses=decision.cited_clauses,
        missing_info=decision.missing_info,
        retrieval_attempts=result.get("r_attempts", 0),
        critique_attempts=result.get("c_attempts", 0),
    )
