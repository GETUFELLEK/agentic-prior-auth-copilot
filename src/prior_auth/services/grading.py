
"""Services for grading retrieved context and reformulating queries."""

from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from prior_auth.schemas import Grade
from prior_auth.services.context import format_policy_documents


def grade_context(
    *,
    question: str,
    documents: list[Document],
    grader: Any,
) -> Grade:
    """
    Determine whether retrieved policy context is relevant and sufficient.

    The grader dependency should be a structured-output model whose
    invoke() method returns a Grade object or compatible dictionary.
    """
    prompt = (
        f"Question:\n{question}\n\n"
        "Policy context:\n"
        f"{format_policy_documents(documents)}\n\n"
        "Determine whether the policy context is both relevant and "
        "sufficient to make a prior-authorization coverage decision.\n"
        "Explain the reason for your assessment."
    )

    raw_result = grader.invoke(
        [HumanMessage(content=prompt)]
    )

    return Grade.model_validate(raw_result)


def reformulate_query(
    *,
    question: str,
    chat_model: Any,
) -> str:
    """
    Rewrite a coverage question into a more precise retrieval query.

    The chat_model dependency should expose an invoke() method returning
    an object with a content attribute.
    """
    prompt = (
        "The first policy retrieval attempt was insufficient.\n\n"
        "Rewrite the following prior-authorization question as a precise "
        "search query for medical coverage policies.\n\n"
        "Include the relevant procedure, diagnosis, test, medical device, "
        "or clinical criteria when they are present in the question.\n\n"
        "Return only the rewritten query.\n\n"
        f"Question:\n{question}"
    )

    response = chat_model.invoke(
        [HumanMessage(content=prompt)]
    )

    content = getattr(response, "content", response)
    reformulated_query = str(content).strip()

    if not reformulated_query:
        raise ValueError(
            "The query reformulation model returned an empty response."
        )

    return reformulated_query
