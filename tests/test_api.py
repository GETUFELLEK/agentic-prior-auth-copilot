
"""Unit tests for the FastAPI application."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from prior_auth.api import create_app


class FakeGraph:
    """Return a fixed completed graph state."""

    def __init__(
        self,
        *,
        result: dict[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.last_state = None
        self.last_config = None

    def invoke(
        self,
        state: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        self.last_state = state
        self.last_config = config

        if self.error is not None:
            raise self.error

        if self.result is None:
            raise AssertionError(
                "FakeGraph requires a result or error."
            )

        return self.result


def approved_result() -> dict[str, Any]:
    return {
        "messages": [
            AIMessage(
                content="DECISION: APPROVE"
            )
        ],
        "question": (
            "Is MRI covered for a patient "
            "with a pacemaker?"
        ),
        "query": "",
        "documents": [],
        "decision": {
            "decision": "APPROVE",
            "cited_clauses": [
                (
                    "MRI is covered when device "
                    "compatibility requirements are met."
                )
            ],
            "rationale": (
                "The submitted request satisfies "
                "the stated policy criteria."
            ),
            "missing_info": None,
        },
        "sufficient": True,
        "grounded": True,
        "grade_reason": "Context was sufficient.",
        "critique_feedback": (
            "The decision is grounded."
        ),
        "human_review_status": "not-required",
        "review_mode": "skip",
        "retrieval_attempts": 1,
        "critique_attempts": 1,
    }


def test_health_endpoint() -> None:
    fake_graph = FakeGraph(
        result=approved_result()
    )

    app = create_app(
        graph_provider=lambda: fake_graph
    )

    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "healthy"
    assert payload["service"]
    assert payload["version"]
    assert payload["environment"]


def test_create_decision_returns_structured_response() -> None:
    fake_graph = FakeGraph(
        result=approved_result()
    )

    app = create_app(
        graph_provider=lambda: fake_graph
    )

    client = TestClient(app)

    response = client.post(
        "/v1/decisions",
        json={
            "question": (
                "Is MRI covered for a patient "
                "with a pacemaker?"
            )
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["decision"] == "APPROVE"
    assert payload["retrieval_attempts"] == 1
    assert payload["critique_attempts"] == 1
    assert (
        payload["human_review_status"]
        == "not-required"
    )
    assert payload["request_id"]

    assert (
        fake_graph.last_state["review_mode"]
        == "skip"
    )

    assert (
        fake_graph.last_config[
            "configurable"
        ]["thread_id"]
        == payload["request_id"]
    )


def test_request_validation_rejects_short_question() -> None:
    fake_graph = FakeGraph(
        result=approved_result()
    )

    app = create_app(
        graph_provider=lambda: fake_graph
    )

    client = TestClient(app)

    response = client.post(
        "/v1/decisions",
        json={
            "question": "MRI",
        },
    )

    assert response.status_code == 422


def test_request_validation_rejects_missing_question() -> None:
    fake_graph = FakeGraph(
        result=approved_result()
    )

    app = create_app(
        graph_provider=lambda: fake_graph
    )

    client = TestClient(app)

    response = client.post(
        "/v1/decisions",
        json={},
    )

    assert response.status_code == 422


def test_workflow_failure_returns_500() -> None:
    fake_graph = FakeGraph(
        error=RuntimeError(
            "Simulated workflow failure."
        )
    )

    app = create_app(
        graph_provider=lambda: fake_graph
    )

    client = TestClient(app)

    response = client.post(
        "/v1/decisions",
        json={
            "question": (
                "Is MRI covered for a patient "
                "with a pacemaker?"
            )
        },
    )

    assert response.status_code == 500

    detail = response.json()["detail"]

    assert (
        detail["message"]
        == "The prior-authorization workflow failed."
    )
    assert detail["request_id"]


def test_invalid_workflow_decision_returns_422() -> None:
    result = approved_result()
    result["decision"] = None

    fake_graph = FakeGraph(
        result=result
    )

    app = create_app(
        graph_provider=lambda: fake_graph
    )

    client = TestClient(app)

    response = client.post(
        "/v1/decisions",
        json={
            "question": (
                "Is MRI covered for a patient "
                "with a pacemaker?"
            )
        },
    )

    assert response.status_code == 422

    detail = response.json()["detail"]

    assert (
        detail["message"]
        == (
            "The workflow could not produce a valid "
            "prior-authorization decision."
        )
    )
