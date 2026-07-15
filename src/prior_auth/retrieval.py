
"""Hybrid dense and BM25 retrieval for medical coverage policies."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from prior_auth.config import get_settings
from prior_auth.llm import get_embedding_model


class HybridPolicyRetriever:
    """Retrieve policy chunks using dense FAISS and sparse BM25 search."""

    def __init__(
        self,
        *,
        dense_k: int = 8,
        sparse_k: int = 8,
    ) -> None:
        if dense_k < 1:
            raise ValueError("dense_k must be at least 1.")

        if sparse_k < 1:
            raise ValueError("sparse_k must be at least 1.")

        self._settings = get_settings()
        self._validate_required_files()

        embeddings = get_embedding_model()

        vector_store = FAISS.load_local(
            str(self._settings.faiss_index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

        self._dense_retriever = vector_store.as_retriever(
            search_kwargs={"k": dense_k}
        )

        policy_documents = self._load_policy_chunks()

        self._sparse_retriever = BM25Retriever.from_documents(
            policy_documents
        )
        self._sparse_retriever.k = sparse_k

    def _validate_required_files(self) -> None:
        """Raise clear errors when policy or FAISS files are missing."""
        policies_dir = self._settings.policies_dir
        faiss_index_dir = self._settings.faiss_index_dir

        if not policies_dir.exists():
            raise RuntimeError(
                f"Policies directory does not exist: {policies_dir}"
            )

        policy_files = sorted(policies_dir.glob("*.pdf"))

        if not policy_files:
            raise RuntimeError(
                f"No PDF policy files were found in: {policies_dir}"
            )

        required_index_files = (
            faiss_index_dir / "index.faiss",
            faiss_index_dir / "index.pkl",
        )

        missing_index_files = [
            path for path in required_index_files if not path.exists()
        ]

        if missing_index_files:
            formatted_paths = "\n".join(
                f"- {path}" for path in missing_index_files
            )

            raise RuntimeError(
                "The FAISS index is incomplete or missing.\n"
                "Run the ingestion script before starting the service.\n"
                f"Missing files:\n{formatted_paths}"
            )

    def _load_policy_chunks(self) -> list[Document]:
        """Load and split policy PDFs for BM25 retrieval."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )

        documents: list[Document] = []

        for policy_path in sorted(
            self._settings.policies_dir.glob("*.pdf")
        ):
            pages = PyPDFLoader(str(policy_path)).load()
            chunks = splitter.split_documents(pages)

            for chunk in chunks:
                chunk.metadata["policy_file"] = policy_path.name

            documents.extend(chunks)

        if not documents:
            raise RuntimeError(
                "Policy files were found, but no text could be extracted."
            )

        return documents

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
    ) -> list[Document]:
        """Return deduplicated policy chunks ranked by hybrid retrieval."""
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("query must not be empty.")

        requested_top_k = (
            top_k
            if top_k is not None
            else self._settings.retrieval_top_k
        )

        if requested_top_k < 1:
            raise ValueError("top_k must be at least 1.")

        dense_documents = self._dense_retriever.invoke(
            normalized_query
        )
        sparse_documents = self._sparse_retriever.invoke(
            normalized_query
        )

        candidates = dense_documents + sparse_documents

        unique_documents: list[Document] = []
        seen_keys: set[str] = set()

        for document in candidates:
            key = self._document_key(document)

            if key in seen_keys:
                continue

            seen_keys.add(key)
            unique_documents.append(document)

            if len(unique_documents) >= requested_top_k:
                break

        return unique_documents

    @staticmethod
    def _document_key(document: Document) -> str:
        """Create a stable deduplication key for a retrieved chunk."""
        policy_file = str(
            document.metadata.get("policy_file", "unknown")
        )
        page = str(document.metadata.get("page", "unknown"))
        content_prefix = document.page_content[:300].strip()

        return f"{policy_file}|{page}|{content_prefix}"


@lru_cache(maxsize=1)
def get_retriever() -> HybridPolicyRetriever:
    """Create the shared retriever only on first use."""
    return HybridPolicyRetriever()


def clear_retriever_cache() -> None:
    """Clear the cached retriever, primarily for tests."""
    get_retriever.cache_clear()


# import glob, os
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.retrievers import BM25Retriever
# # from sentence_transformers import CrossEncoder

# def _build_chunks():
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
#     docs = []
#     for path in glob.glob("policies/*.pdf"):
#         for c in splitter.split_documents(PyPDFLoader(path).load()):
#             c.metadata["policy_file"] = os.path.basename(path)
#             docs.append(c)
#     return docs

# emb = OpenAIEmbeddings(model="text-embedding-3-small")
# _dense = FAISS.load_local("faiss_index", emb,
#                           allow_dangerous_deserialization=True).as_retriever(search_kwargs={"k": 8})
# _bm25 = BM25Retriever.from_documents(_build_chunks()); _bm25.k = 8
# # _reranker = CrossEncoder("BAAI/bge-reranker-base")   # downloads ~1GB on first run

# def hybrid_retrieve(query, top_k=5):
#     pool = _dense.invoke(query) + _bm25.invoke(query)
#     seen, uniq = set(), []
#     for d in pool:
#         key = d.page_content[:200]
#         if key not in seen:
#             seen.add(key); uniq.append(d)
#     return uniq[:top_k]