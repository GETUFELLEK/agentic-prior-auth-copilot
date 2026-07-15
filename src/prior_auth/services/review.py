
"""Human-review logic for denial decisions."""

from collections.abc import Callable

from prior_auth.schemas import PADecision, ReviewMode


InputFunction = Callable[[str], str]
OutputFunction = Callable[[str], None]


def review_denial(
    *,
    decision: PADecision,
    review_mode: ReviewMode,
    input_fn: InputFunction = input,
    output_fn: OutputFunction = print,
) -> str:
    """
    Review a DENY decision.

    Returns one of:
    - not-required
    - skipped
    - approved
    - overridden
    """
    if decision.decision != "DENY":
        return "not-required"

    if review_mode == "skip":
        return "skipped"

    output_fn("")
    output_fn("[HUMAN REVIEW — DENY DECISION]")
    output_fn(f"Rationale: {decision.rationale}")

    if decision.cited_clauses:
        output_fn("Cited clauses:")

        for clause in decision.cited_clauses:
            output_fn(f"- {clause}")

    response = input_fn(
        "Approve this DENY decision? (y/n): "
    ).strip().lower()

    if response == "y":
        return "approved"

    return "overridden"
