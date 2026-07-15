
"""Unit tests for retrieval behavior that do not call external services."""

from langchain_core.documents import Document

from prior_auth.retrieval import HybridPolicyRetriever


def test_document_key_changes_by_policy() -> None:
    first = Document(
        page_content="Coverage requires documented medical necessity.",
        metadata={
            "policy_file": "policy_a.pdf",
            "page": 1,
        },
    )

    second = Document(
        page_content="Coverage requires documented medical necessity.",
        metadata={
            "policy_file": "policy_b.pdf",
            "page": 1,
        },
    )

    first_key = HybridPolicyRetriever._document_key(first)
    second_key = HybridPolicyRetriever._document_key(second)

    assert first_key != second_key


def test_document_key_is_stable() -> None:
    document = Document(
        page_content="MRI coverage criteria.",
        metadata={
            "policy_file": "mri_policy.pdf",
            "page": 2,
        },
    )

    assert (
        HybridPolicyRetriever._document_key(document)
        == HybridPolicyRetriever._document_key(document)
    )
