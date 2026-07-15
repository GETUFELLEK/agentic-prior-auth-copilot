
"""Unit tests for prior-authorization business services."""

from dataclasses import dataclass

from langchain_core.documents import Document

from prior_auth.schemas import Critique, Grade, PADecision
from prior_auth.services.context import format_policy_documents
from prior_auth.services.decision import (
    critique_decision,
    propose_decision,
)
from prior_auth.services.grading import (
    grade_context,
    reformulate_query,
)
from prior_auth.services.review import review_denial


class FakeStructuredModel:
    """Return a fixed structured response from invoke()."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.last_input = None

    def invoke(self, model_input: object) -> object:
        self.last_input = model_input
        return self.response


@dataclass
class FakeMessage:
    content: str


class FakeChatModel:
    """Return a fixed text response from invoke()."""

    def __init__(self, content: str) -> None:
        self.content = content

    def invoke(self, model_input: object) -> FakeMessage:
        return FakeMessage(content=self.content)


def make_document() -> Document:
    return Document(
        page_content=(
            "MRI coverage requires documentation that the implanted "
            "device is MRI conditional."
        ),
        metadata={
            "policy_file": "mri_policy.pdf",
            "page": 1,
        },
    )


def test_format_policy_documents_includes_source() -> None:
    formatted = format_policy_documents(
        [make_document()]
    )

    assert "mri_policy.pdf" in formatted
    assert "page: 2" in formatted
    assert "MRI conditional" in formatted


def test_grade_context_returns_grade() -> None:
    grader = FakeStructuredModel(
        Grade(
            sufficient=True,
            reason="The policy directly addresses the device.",
        )
    )

    result = grade_context(
        question="Is MRI covered with a pacemaker?",
        documents=[make_document()],
        grader=grader,
    )

    assert result.sufficient is True
    assert "directly addresses" in result.reason


def test_reformulate_query_returns_clean_text() -> None:
    chat_model = FakeChatModel(
        "MRI coverage criteria for MRI-conditional pacemakers"
    )

    result = reformulate_query(
        question="Can this patient receive an MRI?",
        chat_model=chat_model,
    )

    assert result == (
        "MRI coverage criteria for MRI-conditional pacemakers"
    )


def test_propose_decision_returns_structured_decision() -> None:
    proposer = FakeStructuredModel(
        PADecision(
            decision="NEEDS_INFO",
            cited_clauses=[
                "The implanted device must be documented as MRI conditional."
            ],
            rationale=(
                "The submitted information does not identify the "
                "pacemaker model."
            ),
            missing_info="Pacemaker model and MRI-conditional status.",
        )
    )

    result = propose_decision(
        question="Is MRI covered with a pacemaker?",
        documents=[make_document()],
        proposer=proposer,
    )

    assert result.decision == "NEEDS_INFO"
    assert result.missing_info is not None


def test_critique_decision_returns_critique() -> None:
    critic = FakeStructuredModel(
        Critique(
            grounded=True,
            reason="The cited clause appears in the retrieved policy.",
        )
    )

    decision = PADecision(
        decision="NEEDS_INFO",
        cited_clauses=[
            "The implanted device must be documented as MRI conditional."
        ],
        rationale="Device compatibility documentation is missing.",
        missing_info="Device model and compatibility status.",
    )

    result = critique_decision(
        decision=decision,
        documents=[make_document()],
        critic=critic,
    )

    assert result.grounded is True


def test_review_denial_skips_in_skip_mode() -> None:
    decision = PADecision(
        decision="DENY",
        cited_clauses=["Coverage criteria were not met."],
        rationale="The submitted request does not meet policy criteria.",
    )

    result = review_denial(
        decision=decision,
        review_mode="skip",
    )

    assert result == "skipped"


def test_review_denial_accepts_interactive_approval() -> None:
    decision = PADecision(
        decision="DENY",
        cited_clauses=["Coverage criteria were not met."],
        rationale="The submitted request does not meet policy criteria.",
    )

    outputs: list[str] = []

    result = review_denial(
        decision=decision,
        review_mode="interactive",
        input_fn=lambda _: "y",
        output_fn=outputs.append,
    )

    assert result == "approved"
    assert any(
        "HUMAN REVIEW" in line
        for line in outputs
    )


def test_review_not_required_for_approval() -> None:
    decision = PADecision(
        decision="APPROVE",
        cited_clauses=["All coverage criteria are satisfied."],
        rationale="The request meets the policy requirements.",
    )

    result = review_denial(
        decision=decision,
        review_mode="interactive",
    )

    assert result == "not-required"
