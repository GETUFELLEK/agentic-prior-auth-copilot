
"""LangGraph orchestration for prior-authorization decisions."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from prior_auth.config import get_settings
from prior_auth.llm import get_chat_model
from prior_auth.retrieval import HybridPolicyRetriever, get_retriever
from prior_auth.schemas import (
    Critique,
    Grade,
    PADecision,
    ReviewMode,
)
from prior_auth.services.decision import (
    critique_decision,
    propose_decision,
)
from prior_auth.services.grading import (
    grade_context,
    reformulate_query,
)
from prior_auth.services.review import review_denial


RouteAfterGrade = Literal["reformulate", "propose"]
RouteAfterCritique = Literal[
    "propose",
    "human_review",
    "finalize",
]


class PriorAuthState(TypedDict):
    """Shared state passed between LangGraph nodes."""

    messages: Annotated[list[BaseMessage], add_messages]

    question: str
    query: str
    documents: list[Document]

    decision: dict[str, Any] | None
    sufficient: bool | None
    grounded: bool | None

    grade_reason: str
    critique_feedback: str
    human_review_status: str
    review_mode: ReviewMode

    retrieval_attempts: int
    critique_attempts: int


def make_initial_state(
    question: str,
    *,
    review_mode: ReviewMode = "skip",
) -> PriorAuthState:
    """Create a fresh state for one prior-authorization request."""
    normalized_question = question.strip()

    if not normalized_question:
        raise ValueError("question must not be empty.")

    return {
        "messages": [
            HumanMessage(content=normalized_question),
        ],
        "question": normalized_question,
        "query": "",
        "documents": [],
        "decision": None,
        "sufficient": None,
        "grounded": None,
        "grade_reason": "",
        "critique_feedback": "",
        "human_review_status": "not-required",
        "review_mode": review_mode,
        "retrieval_attempts": 0,
        "critique_attempts": 0,
    }


def validate_decision(
    raw_decision: object,
) -> PADecision:
    """Convert graph state data into a validated PADecision."""
    if isinstance(raw_decision, PADecision):
        return raw_decision

    if isinstance(raw_decision, dict):
        return PADecision.model_validate(raw_decision)

    raise ValueError(
        "The workflow has not produced a valid decision."
    )


def build_graph(
    *,
    retriever: HybridPolicyRetriever | Any | None = None,
    chat_model: Any | None = None,
    checkpointer: Any | None = None,
):
    """
    Build and compile the prior-authorization workflow.

    Dependencies can be injected for unit tests. When omitted, production
    implementations are created lazily.
    """
    settings = get_settings()

    policy_retriever = (
        retriever
        if retriever is not None
        else get_retriever()
    )

    model = (
        chat_model
        if chat_model is not None
        else get_chat_model()
    )

    grader = model.with_structured_output(Grade)
    proposer = model.with_structured_output(PADecision)
    critic = model.with_structured_output(Critique)

    def retrieve_node(
        state: PriorAuthState,
    ) -> dict[str, Any]:
        """Retrieve policy documents for the current query."""
        query = state["query"] or state["question"]

        documents = policy_retriever.retrieve(
            query,
            top_k=settings.retrieval_top_k,
        )

        return {
            "documents": documents,
            "retrieval_attempts": (
                state["retrieval_attempts"] + 1
            ),
        }

    def grade_node(
        state: PriorAuthState,
    ) -> dict[str, Any]:
        """Grade retrieved context for relevance and sufficiency."""
        result = grade_context(
            question=state["question"],
            documents=state["documents"],
            grader=grader,
        )

        return {
            "sufficient": result.sufficient,
            "grade_reason": result.reason,
        }

    def reformulate_node(
        state: PriorAuthState,
    ) -> dict[str, Any]:
        """Rewrite the question for another retrieval attempt."""
        query = reformulate_query(
            question=state["question"],
            chat_model=model,
        )

        return {
            "query": query,
        }

    def propose_node(
        state: PriorAuthState,
    ) -> dict[str, Any]:
        """Generate a structured policy-grounded decision."""
        decision = propose_decision(
            question=state["question"],
            documents=state["documents"],
            proposer=proposer,
            critique_feedback=state[
                "critique_feedback"
            ],
        )

        return {
            "decision": decision.model_dump(),
            "critique_attempts": (
                state["critique_attempts"] + 1
            ),
        }

    def critique_node(
        state: PriorAuthState,
    ) -> dict[str, Any]:
        """Evaluate whether the proposed decision is grounded."""
        decision = validate_decision(
            state["decision"]
        )

        critique = critique_decision(
            decision=decision,
            documents=state["documents"],
            critic=critic,
        )

        return {
            "grounded": critique.grounded,
            "critique_feedback": critique.reason,
        }

    def human_review_node(
        state: PriorAuthState,
    ) -> dict[str, Any]:
        """Apply interactive or skipped review for a denial."""
        decision = validate_decision(
            state["decision"]
        )

        review_status = review_denial(
            decision=decision,
            review_mode=state["review_mode"],
        )

        return {
            "human_review_status": review_status,
        }

    def finalize_node(
        state: PriorAuthState,
    ) -> dict[str, Any]:
        """Add a readable final decision to message history."""
        decision = validate_decision(
            state["decision"]
        )

        message_parts = [
            f"DECISION: {decision.decision}",
            f"RATIONALE: {decision.rationale}",
        ]

        if decision.cited_clauses:
            message_parts.append(
                "CITED CLAUSES: "
                + "; ".join(decision.cited_clauses)
            )

        if decision.missing_info:
            message_parts.append(
                "MISSING INFORMATION: "
                + decision.missing_info
            )

        return {
            "messages": [
                AIMessage(
                    content="\n".join(message_parts)
                )
            ],
        }

    def route_after_grade(
        state: PriorAuthState,
    ) -> RouteAfterGrade:
        """Choose between another retrieval and decision generation."""
        if state["sufficient"]:
            return "propose"

        if (
            state["retrieval_attempts"]
            < settings.max_retrieval_attempts
        ):
            return "reformulate"

        return "propose"

    def route_after_critique(
        state: PriorAuthState,
    ) -> RouteAfterCritique:
        """Retry unsupported drafts or continue to completion."""
        if (
            state["grounded"] is False
            and state["critique_attempts"]
            < settings.max_critique_attempts
        ):
            return "propose"

        decision = validate_decision(
            state["decision"]
        )

        if decision.decision == "DENY":
            return "human_review"

        return "finalize"

    builder = StateGraph(PriorAuthState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade", grade_node)
    builder.add_node(
        "reformulate",
        reformulate_node,
    )
    builder.add_node("propose", propose_node)
    builder.add_node("critique", critique_node)
    builder.add_node(
        "human_review",
        human_review_node,
    )
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")

    builder.add_conditional_edges(
        "grade",
        route_after_grade,
        {
            "reformulate": "reformulate",
            "propose": "propose",
        },
    )

    builder.add_edge(
        "reformulate",
        "retrieve",
    )

    builder.add_edge(
        "propose",
        "critique",
    )

    builder.add_conditional_edges(
        "critique",
        route_after_critique,
        {
            "propose": "propose",
            "human_review": "human_review",
            "finalize": "finalize",
        },
    )

    builder.add_edge(
        "human_review",
        "finalize",
    )

    builder.add_edge(
        "finalize",
        END,
    )

    graph_checkpointer = (
        checkpointer
        if checkpointer is not None
        else InMemorySaver()
    )

    return builder.compile(
        checkpointer=graph_checkpointer
    )


@lru_cache(maxsize=1)
def get_graph():
    """Build the production graph only when first requested."""
    return build_graph()


def clear_graph_cache() -> None:
    """Clear the cached production graph, primarily for tests."""
    get_graph.cache_clear()


def run_decision(
    *,
    question: str,
    thread_id: str,
    review_mode: ReviewMode = "skip",
) -> PriorAuthState:
    """Execute one prior-authorization request."""
    if not thread_id.strip():
        raise ValueError(
            "thread_id must not be empty."
        )

    graph = get_graph()

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    return graph.invoke(
        make_initial_state(
            question,
            review_mode=review_mode,
        ),
        config,
    )

