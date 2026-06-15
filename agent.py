import os
from typing import TypedDict, Annotated, List, Literal, Optional
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from retrieval import hybrid_retrieve
from models import Grade, PADecision, Critique

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
MAX_RETRIEVE, MAX_CRITIQUE = 2, 2

# class Grade(BaseModel):
#     sufficient: bool = Field(description="context is sufficient AND relevant to decide")
#     reason: str

# class PADecision(BaseModel):
#     decision: Literal["APPROVE", "DENY", "NEEDS_INFO"]
#     cited_clauses: List[str]
#     rationale: str
#     missing_info: Optional[str] = None

# class Critique(BaseModel):
#     grounded: bool = Field(description="every claim is supported by cited context, no hallucination")
#     reason: str
class State(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    query: str
    documents: List[Document]
    decision: Optional[PADecision]
    sufficient: Optional[bool]
    grounded: Optional[bool]
    critique_feedback: str
    r_attempts: int
    c_attempts: int

def _fmt(docs):
    return "\n\n---\n\n".join(f"[{d.metadata.get('policy_file','?')}] {d.page_content}" for d in docs)

grader   = llm.with_structured_output(Grade)
proposer = llm.with_structured_output(PADecision)
critic   = llm.with_structured_output(Critique)

def retrieve(state):
    q = state["query"] or state["question"]
    return {"documents": hybrid_retrieve(q, top_k=5), "r_attempts": state["r_attempts"] + 1}

def grade(state):
    g = grader.invoke([HumanMessage(content=
        f"Question: {state['question']}\n\nPolicy context:\n{_fmt(state['documents'])}\n\n"
        "Is this context sufficient and relevant to make a coverage decision?")])
    return {"sufficient": g.sufficient}

def reformulate(state):
    q = llm.invoke([HumanMessage(content=
        "Retrieval was insufficient. Rewrite this as a more precise search query over medical "
        f"coverage policies. Return only the query.\n\nQuestion: {state['question']}")]).content
    return {"query": q}

def propose(state):
    fb = state["critique_feedback"]
    hist = "\n".join(f"{m.type}: {m.content}" for m in state["messages"][:-1])[-1500:]
    parts = ["You are a prior-authorization assistant. Decide using ONLY the policy context.",
             "Cite exact clauses. If context is insufficient, choose NEEDS_INFO and fill missing_info."]
    if hist: parts.append(f"Conversation so far:\n{hist}")
    if fb and fb not in ("human-approved", "human-overridden"):
        parts.append(f"Reviewer feedback to address:\n{fb}")
    parts += [f"Question: {state['question']}", f"Policy context:\n{_fmt(state['documents'])}"]
    return {"decision": proposer.invoke([HumanMessage(content="\n\n".join(parts))]).model_dump(),
            "c_attempts": state["c_attempts"] + 1}

def critique(state):
    d = state["decision"]
    c = critic.invoke([HumanMessage(content=
        "Strictly check the draft against the policy context. Are the cited clauses actually present, "
        "and does the decision follow the criteria? Flag any unsupported claim.\n\n"
        f"Policy context:\n{_fmt(state['documents'])}\n\n"
        f"Draft: {d.decision}\nCited: {d.cited_clauses}\nRationale: {d.rationale}")])
    return {"grounded": c.grounded, "critique_feedback": c.reason}



def human_review(state):
    d = state["decision"]
    if os.getenv("EVAL_MODE") == "1":                 # batch eval: no prompt, log + pass through
        return {"critique_feedback": "auto-approved (eval mode)"}
    rationale = d["rationale"] if isinstance(d, dict) else d.rationale
    print("\n  [HUMAN REVIEW — DENY decision]")
    print("  Rationale:", rationale)
    ok = input("  Approve this DENY? (y/n): ").strip().lower()
    return {"critique_feedback": "human-approved" if ok == "y" else "human-overridden"}

def finalize(state):
    d = state["decision"]
    return {"messages": [AIMessage(content=f"DECISION: {d.decision} — {d.rationale}")]}

def after_grade(state):
    if state["sufficient"]: return "propose"
    return "reformulate" if state["r_attempts"] < MAX_RETRIEVE else "propose"

def after_critique(state):
    if not state["grounded"] and state["c_attempts"] < MAX_CRITIQUE:
        return "propose"
    return "human_review" if state["decision"].decision == "DENY" else "finalize"

g = StateGraph(State)
for n, f in [("retrieve", retrieve), ("grade", grade), ("reformulate", reformulate),
             ("propose", propose), ("critique", critique),
             ("human_review", human_review), ("finalize", finalize)]:
    g.add_node(n, f)
g.add_edge(START, "retrieve")
g.add_edge("retrieve", "grade")
g.add_conditional_edges("grade", after_grade, {"propose": "propose", "reformulate": "reformulate"})
g.add_edge("reformulate", "retrieve")
g.add_edge("propose", "critique")
g.add_conditional_edges("critique", after_critique,
                        {"propose": "propose", "human_review": "human_review", "finalize": "finalize"})
g.add_edge("human_review", "finalize")
g.add_edge("finalize", END)
app = g.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    cfg = {"configurable": {"thread_id": "demo"}}
    print("Prior-auth copilot — ask a coverage question (Ctrl-C to exit).")
    while True:
        q = input("\nQ: ").strip()
        if not q: continue
        out = app.invoke({"messages": [HumanMessage(content=q)], "question": q, "query": "",
                          "documents": [], "decision": None, "sufficient": None, "grounded": None,
                          "critique_feedback": "", "r_attempts": 0, "c_attempts": 0}, cfg)
        d = out["decision"]
        print(f"\nDECISION: {d.decision}\nRationale: {d.rationale}")
        if d.cited_clauses: print("Cited:", "; ".join(d.cited_clauses))
        if d.missing_info:  print("Missing:", d.missing_info)