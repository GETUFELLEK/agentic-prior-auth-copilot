
import os
from typing import Annotated, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from models import Critique, Grade, PADecision
from retrieval import hybrid_retrieve


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

MAX_RETRIEVE = 2
MAX_CRITIQUE = 2

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0,
)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class State(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    query: str
    documents: List[Document]
    decision: Optional[dict]
    sufficient: Optional[bool]
    grounded: Optional[bool]
    critique_feedback: str
    r_attempts: int
    c_attempts: int


# ---------------------------------------------------------------------------
# Structured-output LLMs
# ---------------------------------------------------------------------------

grader = llm.with_structured_output(Grade)
proposer = llm.with_structured_output(PADecision)
critic = llm.with_structured_output(Critique)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def make_initial_state(question: str) -> State:
    """Create a fresh LangGraph state for one prior-authorization request."""
    return {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "query": "",
        "documents": [],
        "decision": None,
        "sufficient": None,
        "grounded": None,
        "critique_feedback": "",
        "r_attempts": 0,
        "c_attempts": 0,
    }


def _format_documents(documents: List[Document]) -> str:
    """Format retrieved policy documents for use in an LLM prompt."""
    if not documents:
        return "No policy context was retrieved."

    return "\n\n---\n\n".join(
        (
            f"[Policy: {document.metadata.get('policy_file', 'unknown')}]\n"
            f"{document.page_content}"
        )
        for document in documents
    )


def _validate_decision(raw_decision) -> PADecision:
    """Convert either a dictionary or PADecision object into PADecision."""
    if isinstance(raw_decision, PADecision):
        return raw_decision

    if isinstance(raw_decision, dict):
        return PADecision.model_validate(raw_decision)

    raise ValueError(
        f"Unsupported decision type: {type(raw_decision).__name__}"
    )


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def retrieve(state: State) -> dict:
    """Retrieve relevant policy chunks using hybrid dense and BM25 search."""
    query = state["query"] or state["question"]

    documents = hybrid_retrieve(
        query=query,
        top_k=5,
    )

    return {
        "documents": documents,
        "r_attempts": state["r_attempts"] + 1,
    }


def grade(state: State) -> dict:
    """Determine whether the retrieved context is sufficient and relevant."""
    prompt = (
        f"Question:\n{state['question']}\n\n"
        f"Policy context:\n{_format_documents(state['documents'])}\n\n"
        "Determine whether this context is both relevant and sufficient "
        "to make a prior-authorization coverage decision."
    )

    result = grader.invoke(
        [HumanMessage(content=prompt)]
    )

    return {
        "sufficient": result.sufficient,
    }


def reformulate(state: State) -> dict:
    """Rewrite the user's question into a more precise retrieval query."""
    prompt = (
        "The retrieved medical coverage-policy context was insufficient.\n\n"
        "Rewrite the following question as a precise search query for medical "
        "coverage policies. Include the procedure, diagnosis, device, test, "
        "or clinical criteria that are important.\n\n"
        "Return only the rewritten search query.\n\n"
        f"Question:\n{state['question']}"
    )

    response = llm.invoke(
        [HumanMessage(content=prompt)]
    )

    return {
        "query": response.content.strip(),
    }


def propose(state: State) -> dict:
    """Generate a structured prior-authorization decision."""
    conversation_history = "\n".join(
        f"{message.type}: {message.content}"
        for message in state["messages"][:-1]
    )

    conversation_history = conversation_history[-1500:]

    prompt_parts = [
        (
            "You are a prior-authorization decision-support assistant.\n"
            "Make a decision using only the supplied policy context.\n"
            "Do not invent clinical facts or policy requirements."
        ),
        (
            "Return one of these decisions:\n"
            "- APPROVE\n"
            "- DENY\n"
            "- NEEDS_INFO"
        ),
        (
            "Cite the exact relevant policy language in cited_clauses.\n"
            "If the policy context or patient information is insufficient, "
            "choose NEEDS_INFO and clearly state what information is missing."
        ),
    ]

    if conversation_history:
        prompt_parts.append(
            f"Conversation history:\n{conversation_history}"
        )

    feedback = state["critique_feedback"]

    if feedback and feedback not in {
        "human-approved",
        "human-overridden",
        "human-review-skipped",
    }:
        prompt_parts.append(
            f"Reviewer feedback that must be addressed:\n{feedback}"
        )

    prompt_parts.extend(
        [
            f"Question:\n{state['question']}",
            f"Policy context:\n{_format_documents(state['documents'])}",
        ]
    )

    result = proposer.invoke(
        [HumanMessage(content="\n\n".join(prompt_parts))]
    )

    return {
        "decision": result.model_dump(),
        "c_attempts": state["c_attempts"] + 1,
    }


def critique(state: State) -> dict:
    """Check whether the proposed decision is grounded in policy context."""
    decision = _validate_decision(state["decision"])

    prompt = (
        "Strictly review the draft prior-authorization decision against the "
        "provided policy context.\n\n"
        "Check all of the following:\n"
        "1. The cited clauses are actually present in the policy context.\n"
        "2. The rationale follows from the cited policy language.\n"
        "3. The decision does not invent patient facts or requirements.\n"
        "4. Unsupported claims are identified.\n\n"
        f"Policy context:\n{_format_documents(state['documents'])}\n\n"
        f"Draft decision: {decision.decision}\n"
        f"Cited clauses: {decision.cited_clauses}\n"
        f"Rationale: {decision.rationale}\n"
        f"Missing information: {decision.missing_info}"
    )

    result = critic.invoke(
        [HumanMessage(content=prompt)]
    )

    return {
        "grounded": result.grounded,
        "critique_feedback": result.reason,
    }


def human_review(state: State) -> dict:
    """Request terminal review for DENY decisions outside evaluation/API mode."""
    decision = _validate_decision(state["decision"])

    if os.getenv("EVAL_MODE") == "1":
        return {
            "critique_feedback": "human-review-skipped",
        }

    print("\n[HUMAN REVIEW — DENY DECISION]")
    print("Rationale:", decision.rationale)

    if decision.cited_clauses:
        print("Cited clauses:")

        for clause in decision.cited_clauses:
            print("-", clause)

    answer = input(
        "\nApprove this DENY decision? (y/n): "
    ).strip().lower()

    if answer == "y":
        return {
            "critique_feedback": "human-approved",
        }

    return {
        "critique_feedback": "human-overridden",
    }


def finalize(state: State) -> dict:
    """Add the final structured decision to the conversation messages."""
    decision = _validate_decision(state["decision"])

    message = (
        f"DECISION: {decision.decision}\n"
        f"RATIONALE: {decision.rationale}"
    )

    if decision.cited_clauses:
        message += (
            "\nCITED CLAUSES: "
            + "; ".join(decision.cited_clauses)
        )

    if decision.missing_info:
        message += (
            f"\nMISSING INFORMATION: {decision.missing_info}"
        )

    return {
        "messages": [AIMessage(content=message)],
    }


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def after_grade(state: State) -> str:
    """Choose whether to decide or reformulate and retrieve again."""
    if state["sufficient"]:
        return "propose"

    if state["r_attempts"] < MAX_RETRIEVE:
        return "reformulate"

    return "propose"


def after_critique(state: State) -> str:
    """Retry unsupported decisions or route to review/finalization."""
    if not state["grounded"] and state["c_attempts"] < MAX_CRITIQUE:
        return "propose"

    decision = _validate_decision(state["decision"])

    if decision.decision == "DENY":
        return "human_review"

    return "finalize"


# ---------------------------------------------------------------------------
# Build and compile the LangGraph workflow
# ---------------------------------------------------------------------------

graph = StateGraph(State)

graph.add_node("retrieve", retrieve)
graph.add_node("grade", grade)
graph.add_node("reformulate", reformulate)
graph.add_node("propose", propose)
graph.add_node("critique", critique)
graph.add_node("human_review", human_review)
graph.add_node("finalize", finalize)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "grade")

graph.add_conditional_edges(
    "grade",
    after_grade,
    {
        "propose": "propose",
        "reformulate": "reformulate",
    },
)

graph.add_edge("reformulate", "retrieve")
graph.add_edge("propose", "critique")

graph.add_conditional_edges(
    "critique",
    after_critique,
    {
        "propose": "propose",
        "human_review": "human_review",
        "finalize": "finalize",
    },
)

graph.add_edge("human_review", "finalize")
graph.add_edge("finalize", END)

app = graph.compile(
    checkpointer=MemorySaver()
)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the prior-authorization copilot interactively."""
    print(
        "Prior-auth copilot — ask a coverage question "
        "(press Ctrl-C to exit)."
    )

    question_number = 0

    while True:
        try:
            question = input("\nQ: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not question:
            continue

        question_number += 1

        config = {
            "configurable": {
                "thread_id": f"cli-{question_number}",
            }
        }

        try:
            result = app.invoke(
                make_initial_state(question),
                config,
            )

            decision = _validate_decision(result["decision"])

            print(f"\nDECISION: {decision.decision}")
            print(f"Rationale: {decision.rationale}")

            if decision.cited_clauses:
                print("Cited clauses:")

                for clause in decision.cited_clauses:
                    print("-", clause)

            if decision.missing_info:
                print(
                    "Missing information:",
                    decision.missing_info,
                )

        except Exception as exc:
            print(
                f"\nThe prior-authorization workflow failed: {exc}"
            )


if __name__ == "__main__":
    main()
