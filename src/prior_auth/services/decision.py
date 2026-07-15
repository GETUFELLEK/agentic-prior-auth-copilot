
"""Services for proposing and critiquing prior-authorization decisions."""

from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from prior_auth.schemas import Critique, PADecision
from prior_auth.services.context import format_policy_documents


IGNORED_CRITIQUE_FEEDBACK = {
    "",
    "human-approved",
    "human-overridden",
    "human-review-skipped",
}


def propose_decision(
    *,
    question: str,
    documents: list[Document],
    proposer: Any,
    critique_feedback: str = "",
) -> PADecision:
    """
    Produce a policy-grounded structured prior-authorization decision.

    The proposer dependency should be a structured-output model whose
    invoke() method returns a PADecision object or compatible dictionary.
    """
    prompt_parts = [
        (
            "You are a prior-authorization decision-support assistant. "
            "Use only the supplied policy context."
        ),
        (
            "Do not invent patient facts, clinical findings, policy "
            "requirements, or coverage criteria."
        ),
        (
            "Return exactly one decision: APPROVE, DENY, or NEEDS_INFO."
        ),
        (
            "Include exact relevant policy language in cited_clauses. "
            "If the policy context or patient information is insufficient, "
            "choose NEEDS_INFO and identify what information is missing."
        ),
    ]

    normalized_feedback = critique_feedback.strip()

    if normalized_feedback not in IGNORED_CRITIQUE_FEEDBACK:
        prompt_parts.append(
            "Previous critic feedback that must be addressed:\n"
            f"{normalized_feedback}"
        )

    prompt_parts.extend(
        [
            f"Question:\n{question}",
            (
                "Policy context:\n"
                f"{format_policy_documents(documents)}"
            ),
        ]
    )

    raw_result = proposer.invoke(
        [HumanMessage(content="\n\n".join(prompt_parts))]
    )

    return PADecision.model_validate(raw_result)


def critique_decision(
    *,
    decision: PADecision,
    documents: list[Document],
    critic: Any,
) -> Critique:
    """
    Check whether a proposed decision is grounded in the policy context.

    The critic dependency should be a structured-output model whose
    invoke() method returns a Critique object or compatible dictionary.
    """
    prompt = (
        "Strictly review the proposed prior-authorization decision "
        "against the supplied policy context.\n\n"
        "Verify all of the following:\n"
        "1. Every cited clause appears in the policy context.\n"
        "2. The rationale follows from the cited policy language.\n"
        "3. No patient facts or policy requirements were invented.\n"
        "4. The selected decision is supported by the policy context.\n\n"
        "Policy context:\n"
        f"{format_policy_documents(documents)}\n\n"
        f"Decision: {decision.decision}\n"
        f"Cited clauses: {decision.cited_clauses}\n"
        f"Rationale: {decision.rationale}\n"
        f"Missing information: {decision.missing_info}"
    )

    raw_result = critic.invoke(
        [HumanMessage(content=prompt)]
    )

    return Critique.model_validate(raw_result)
