
"""FastAPI application for the prior-authorization copilot."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, HTTPException, status

from prior_auth.config import get_settings
from prior_auth.graph import (
    PriorAuthState,
    get_graph,
    make_initial_state,
    validate_decision,
)
from prior_auth.schemas import (
    DecisionRequest,
    DecisionResponse,
    HealthResponse,
)


logger = logging.getLogger(__name__)

GraphProvider = Callable[[], Any]


def create_app(
    *,
    graph_provider: GraphProvider = get_graph,
) -> FastAPI:
    """
    Create and configure the FastAPI application.

    graph_provider is injectable so API tests can use a fake graph
    without loading FAISS or calling OpenAI.
    """
    settings = get_settings()

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Policy-grounded prior-authorization decision support "
            "using hybrid retrieval, LangGraph orchestration, "
            "structured decision generation, critique, and "
            "human-review routing."
        ),
    )

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["Operations"],
        summary="Check service health",
    )
    def health() -> HealthResponse:
        """
        Return lightweight process health.

        This endpoint deliberately does not initialize OpenAI or FAISS.
        """
        return HealthResponse(
            status="healthy",
            service=settings.app_name,
            version=settings.app_version,
            environment=settings.environment,
        )

    @application.post(
        "/v1/decisions",
        response_model=DecisionResponse,
        status_code=status.HTTP_200_OK,
        tags=["Prior Authorization"],
        summary="Generate a prior-authorization decision",
    )
    def create_decision(
        request: DecisionRequest,
    ) -> DecisionResponse:
        """
        Execute the prior-authorization workflow for one question.

        API requests use review_mode='skip' because an HTTP request
        cannot pause for terminal input. A persistent human-review
        workflow will be introduced separately.
        """
        request_id = str(uuid.uuid4())

        config = {
            "configurable": {
                "thread_id": request_id,
            }
        }

        try:
            graph = graph_provider()

            result: PriorAuthState = graph.invoke(
                make_initial_state(
                    request.question,
                    review_mode="skip",
                ),
                config,
            )

            decision = validate_decision(
                result.get("decision")
            )

        except ValueError as exc:
            logger.warning(
                "Invalid prior-authorization workflow result. "
                "request_id=%s error=%s",
                request_id,
                exc,
            )

            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": (
                        "The workflow could not produce a valid "
                        "prior-authorization decision."
                    ),
                    "request_id": request_id,
                },
            ) from exc

        except Exception as exc:
            logger.exception(
                "Prior-authorization workflow failed. request_id=%s",
                request_id,
            )

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "message": (
                        "The prior-authorization workflow failed."
                    ),
                    "request_id": request_id,
                },
            ) from exc

        return DecisionResponse(
            request_id=request_id,
            decision=decision.decision,
            rationale=decision.rationale,
            cited_clauses=decision.cited_clauses,
            missing_info=decision.missing_info,
            retrieval_attempts=result.get(
                "retrieval_attempts",
                0,
            ),
            critique_attempts=result.get(
                "critique_attempts",
                0,
            ),
            human_review_status=result.get(
                "human_review_status",
                "not-required",
            ),
        )

    return application


api = create_app()
