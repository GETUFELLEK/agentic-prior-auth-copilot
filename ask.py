from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

emb = OpenAIEmbeddings(model="text-embedding-3-small")
store = FAISS.load_local("faiss_index", emb, allow_dangerous_deserialization=True)
retriever = store.as_retriever(search_kwargs={"k": 6})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

PROMPT = ChatPromptTemplate.from_template(
"""You are a prior-authorization assistant. Decide using ONLY the policy context.
Return: DECISION (APPROVE / DENY / NEEDS_INFO), the exact policy clauses you relied on,
and a one-paragraph rationale. If the context is insufficient, return NEEDS_INFO and say what is missing.

Question: {q}

Policy context:
{context}""")

def ask(q):
    ctx = "\n\n---\n\n".join(
        f"[{d.metadata['policy_file']}] {d.page_content}" for d in retriever.invoke(q))
    return (PROMPT | llm).invoke({"q": q, "context": ctx}).content

if __name__ == "__main__":
    print(ask("Is an MRI covered for a patient with a cardiac pacemaker?"))