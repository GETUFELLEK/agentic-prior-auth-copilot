
"""Unit tests for LangGraph orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from prior_auth.graph import (
    build_graph,
    make_initial_state,
    validate_decision,
)
from prior_auth.schemas import (
    Critique,
    Grade,
    PADecision,
)


class FakeRetriever:
    """Return a fixed set of policy documents."""

    def __init__(
        self,
        documents: list[Document],
    ) -> None:
        self.documents = documents
        self.queries: list[str] = []

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[Document]:
        self.queries.append(query)
        return self.documents[:top_k]


class FakeStructuredModel:
    """Return queued structured responses."""

    def __init__(
        self,
        responses: list[object],
    ) -> None:
        self.responses = responses

    def invoke(
        self,
        model_input: object,
    ) -> object:
        if not self.responses:
            raise AssertionError(
                "No fake structured responses remain."
            )

        return self.responses.pop(0)


@dataclass
class FakeTextResponse:
    content: str


class FakeChatModel:
    """Provide structured models and text responses."""

    def __init__(
        self,
        *,
        grades: list[Grade],
        decisions: list[PADecision],
        critiques: list[Critique],
        reformulations: list[str] | None = None,
    ) -> None:
        self._structured_models = {
            Grade: FakeStructuredModel(grades),
            PADecision: FakeStructuredModel(
                decisions
            ),
            Critique: FakeStructuredModel(
                critiques
            ),
        }

        self._reformulations = (
            reformulations or []
        )

    def with_structured_output(
        self,
        schema: type,
    ) -> FakeStructuredModel:
        return self._structured_models[schema]

    def invoke(
        self,
        model_input: object,
    ) -> FakeTextResponse:
        if not self._reformulations:
            raise AssertionError(
                "No fake reformulation responses remain."
            )

        return FakeTextResponse(
            content=self._reformulations.pop(0)
        )


def policy_document() -> Document:
    return Document(
        page_content=(
            "MRI coverage requires documentation "
            "that an implanted cardiac device is "
            "MRI conditional."
        ),
        metadata={
            "policy_file": "mri_policy.pdf",
            "page": 0,
        },
    )


def invoke_graph(
    graph,
    *,
    question: str = (
        "Is MRI covered for a patient "
        "with a pacemaker?"
    ),
):
    return graph.invoke(
        make_initial_state(
            question,
            review_mode="skip",
        ),
        {
            "configurable": {
                "thread_id": "test-thread",
            }
        },
    )


def test_make_initial_state() -> None:
    state = make_initial_state(
        "Is MRI covered?"
    )

    assert state["question"] == (
        "Is MRI covered?"
    )
    assert state["retrieval_attempts"] == 0
    assert state["critique_attempts"] == 0
    assert state["documents"] == []


def test_validate_decision_from_dictionary() -> None:
    decision = validate_decision(
        {
            "decision": "APPROVE",
            "cited_clauses": [
                "Coverage criteria are met."
            ],
            "rationale": (
                "The request satisfies the policy."
            ),
            "missing_info": None,
        }
    )

    assert decision.decision == "APPROVE"


def test_graph_approves_grounded_decision() -> None:
    retriever = FakeRetriever(
        [policy_document()]
    )

    model = FakeChatModel(
        grades=[
            Grade(
                sufficient=True,
                reason=(
                    "The policy directly addresses MRI."
                ),
            )
        ],
        decisions=[
            PADecision(
                decision="APPROVE",
                cited_clauses=[
                    (
                        "MRI coverage requires "
                        "documentation that the "
                        "device is MRI conditional."
                    )
                ],
                rationale=(
                    "The request meets the stated "
                    "coverage requirement."
                ),
            )
        ],
        critiques=[
            Critique(
                grounded=True,
                reason=(
                    "The decision follows the policy."
                ),
            )
        ],
    )

    graph = build_graph(
        retriever=retriever,
        chat_model=model,
    )

    result = invoke_graph(graph)

    decision = validate_decision(
        result["decision"]
    )

    assert decision.decision == "APPROVE"
    assert result["retrieval_attempts"] == 1
    assert result["critique_attempts"] == 1
    assert (
        result["human_review_status"]
        == "not-required"
    )
    assert isinstance(
        result["messages"][-1],
        AIMessage,
    )


def test_graph_reformulates_after_insufficient_context() -> None:
    retriever = FakeRetriever(
        [policy_document()]
    )

    model = FakeChatModel(
        grades=[
            Grade(
                sufficient=False,
                reason="The first result is incomplete.",
            ),
            Grade(
                sufficient=True,
                reason=(
                    "The second query retrieved "
                    "the needed criteria."
                ),
            ),
        ],
        reformulations=[
            (
                "MRI coverage criteria for "
                "MRI-conditional cardiac pacemakers"
            )
        ],
        decisions=[
            PADecision(
                decision="NEEDS_INFO",
                cited_clauses=[
                    (
                        "Device compatibility "
                        "documentation is required."
                    )
                ],
                rationale=(
                    "The pacemaker model was not supplied."
                ),
                missing_info=(
                    "Pacemaker model and "
                    "MRI-conditional status."
                ),
            )
        ],
        critiques=[
            Critique(
                grounded=True,
                reason=(
                    "The missing-information decision "
                    "is supported."
                ),
            )
        ],
    )

    graph = build_graph(
        retriever=retriever,
        chat_model=model,
    )

    result = invoke_graph(graph)

    assert result["retrieval_attempts"] == 2

    assert retriever.queries == [
        (
            "Is MRI covered for a patient "
            "with a pacemaker?"
        ),
        (
            "MRI coverage criteria for "
            "MRI-conditional cardiac pacemakers"
        ),
    ]


def test_graph_retries_after_failed_critique() -> None:
    retriever = FakeRetriever(
        [policy_document()]
    )

    model = FakeChatModel(
        grades=[
            Grade(
                sufficient=True,
                reason="Context is sufficient.",
            )
        ],
        decisions=[
            PADecision(
                decision="DENY",
                cited_clauses=[
                    "Unsupported clause."
                ],
                rationale=(
                    "The request is not covered."
                ),
            ),
            PADecision(
                decision="NEEDS_INFO",
                cited_clauses=[
                    (
                        "Device compatibility "
                        "documentation is required."
                    )
                ],
                rationale=(
                    "Compatibility information "
                    "was not provided."
                ),
                missing_info=(
                    "Pacemaker model and "
                    "MRI compatibility."
                ),
            ),
        ],
        critiques=[
            Critique(
                grounded=False,
                reason=(
                    "The first cited clause "
                    "does not appear in context."
                ),
            ),
            Critique(
                grounded=True,
                reason=(
                    "The revised decision is grounded."
                ),
            ),
        ],
    )

    graph = build_graph(
        retriever=retriever,
        chat_model=model,
    )

    result = invoke_graph(graph)

    decision = validate_decision(
        result["decision"]
    )

    assert decision.decision == "NEEDS_INFO"
    assert result["critique_attempts"] == 2
    assert result["grounded"] is True


def test_graph_skips_human_review_for_denial() -> None:
    retriever = FakeRetriever(
        [policy_document()]
    )

    model = FakeChatModel(
        grades=[
            Grade(
                sufficient=True,
                reason="Context is sufficient.",
            )
        ],
        decisions=[
            PADecision(
                decision="DENY",
                cited_clauses=[
                    "The stated criteria are not met."
                ],
                rationale=(
                    "The request does not satisfy "
                    "the coverage requirement."
                ),
            )
        ],
        critiques=[
            Critique(
                grounded=True,
                reason=(
                    "The denial follows the policy."
                ),
            )
        ],
    )

    graph = build_graph(
        retriever=retriever,
        chat_model=model,
    )

    result = invoke_graph(graph)

    assert result["human_review_status"] == (
        "skipped"
    )
