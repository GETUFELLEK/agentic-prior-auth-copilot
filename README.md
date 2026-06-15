# Agentic Prior-Authorization Copilot

A policy-grounded **agentic RAG** system that answers medical prior-authorization questions over payer/CMS coverage policies. Given a clinical question, it retrieves the governing policy criteria, reasons over them, and returns a structured **`APPROVE` / `DENY` / `NEEDS_INFO`** decision with the exact policy clauses it relied on and a plain-language rationale.

Built to explore the engineering patterns behind reliable, auditable LLM decision-making in a regulated domain — where a wrong action has real consequences, and "I'm not sure" must be a first-class answer.

---

## Why this design

A naive RAG bot pattern-matches "MRI + cancer" and confidently approves. This system is built so that:

- **decisions are grounded** in retrieved policy text, with citations — never free-floating model opinion;
- **insufficient evidence yields `NEEDS_INFO`**, not a hallucinated answer;
- **high-stakes denials are gated** behind human review;
- **every run is traceable** and the system is **evaluated at the decision level**, not just on answer fluency.

---

## Architecture

```
question + memory
      │
      ▼
  hybrid retrieve (dense FAISS + BM25)  ──►  grade relevance
      ▲                                          │
      │ reformulate query (self-correct)  ◄──────┤ insufficient
      │                                          ▼ sufficient
      │                              proposer: draft decision + citations
      │                                          │
      │                              critic/judge: grounded? criteria met?
      │                                  │ fails  │ passes
      │                                  └────────┤
      │                                           ▼
      │                                   DENY ──► human-in-the-loop gate
      │                                   else ──► finalize ──► memory + trace
```

Two control loops make this **agentic**, not just retrieval-augmented:

1. **Grade → reformulate loop** — when retrieved context is judged insufficient, the agent rewrites the query and retrieves again (bounded retries).
2. **Proposer → critic loop** — a draft decision is checked by a separate critic pass for grounding and criteria-fit before it can be returned; ungrounded drafts are sent back for revision.

A **human-in-the-loop checkpoint** intercepts `DENY` decisions for review (configurable: interactive in normal use, auto-bypassed in batch evaluation).

---

## Stack

| Concern | Choice |
|---|---|
| Orchestration | LangGraph (stateful graph, checkpointed memory) |
| Retrieval | Hybrid: FAISS (dense) + BM25 (sparse), with dedupe |
| LLM | OpenAI (provider-agnostic graph) |
| Embeddings | `text-embedding-3-small` |
| Memory | Conversation state via LangGraph checkpointer |
| Observability | LangSmith tracing (per-node traces, prompts, latency) |
| Evaluation | Decision-accuracy harness + RAGAS (retrieval/answer quality) |

---

## Evaluation

Decisions are scored against a hand-labeled question set with ground-truth `APPROVE/DENY/NEEDS_INFO` and the justifying policy section.

- **Decision accuracy** — does the agent's decision match the label? (domain-critical metric)
- **RAGAS** — faithfulness, answer relevancy, context precision/recall (retrieval & generation quality)

Run:

```bash
python ingest.py            # build the local vector index from policies/
EVAL_MODE=1 python eval_decision.py   # scored decision table
```

*(Eval results table and failure-mode analysis: see `/results`.)*

---

## Repo layout

| File | Role |
|---|---|
| `ingest.py` | Load policy PDFs → chunk → embed → FAISS index |
| `retrieval.py` | Hybrid dense + BM25 retrieval |
| `models.py` | Pydantic schemas for structured decisions |
| `agent.py` | LangGraph agent: grade / propose / critic / HITL / memory |
| `eval_set.py` | Hand-labeled ground-truth questions |
| `eval_decision.py` | Decision-accuracy scorer |
| `ask.py` | Minimal retrieve-and-answer baseline |

---

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install langchain langchain-openai langchain-community langgraph \
            faiss-cpu pypdf tiktoken rank-bm25

# secrets via a gitignored .env (never committed)
#   OPENAI_API_KEY=...
#   LANGSMITH_API_KEY=...   (optional, for tracing)

# add 8-12 public CMS coverage PDFs to ./policies/  (see policies/README)
python ingest.py
python agent.py
```

---

## Scope & honesty notes

- Uses **public CMS coverage policies only** — no PHI, by design.
- A research/prototype exploring agentic-RAG reliability patterns, not a clinical product.
- The human-in-the-loop gate is a dev-mode `input()` prompt; a production version would use LangGraph's async `interrupt()` with a durable checkpointer.

---

## What this demonstrates

End-to-end agentic system design — orchestration, hybrid retrieval, grounding, self-correction, human oversight, memory, observability, and decision-level evaluation — applied to high-stakes, policy-grounded decisioning.
