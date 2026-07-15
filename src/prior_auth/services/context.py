"""Helpers for formatting retrieved policy documents."""

from langchain_core.documents import Document


def format_policy_documents(documents: list[Document]) -> str:
    """Format retrieved policy chunks for use in LLM prompts."""
    if not documents:
        return "No policy context was retrieved."

    formatted_documents: list[str] = []

    for document in documents:
        policy_file = document.metadata.get(
            "policy_file",
            "unknown",
        )
        page = document.metadata.get("page")

        source = f"Policy: {policy_file}"

        if isinstance(page, int):
            source += f", page: {page + 1}"

        formatted_documents.append(
            f"[{source}]\n{document.page_content.strip()}"
        )

    return "\n\n---\n\n".join(formatted_documents)
