
"""Lazy construction of OpenAI-backed language and embedding models."""

from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from prior_auth.config import get_settings


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI:
    """
    Return the shared chat model.

    The OpenAI client is initialized only when this function is called,
    not when the module is imported.
    """
    settings = get_settings()

    return ChatOpenAI(
        model=settings.openai_model,
        temperature=0,
        api_key=settings.require_openai_api_key(),
    )


@lru_cache(maxsize=1)
def get_embedding_model() -> OpenAIEmbeddings:
    """
    Return the shared embedding model.

    The embedding client is initialized only when this function is called.
    """
    settings = get_settings()

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.require_openai_api_key(),
    )


def clear_model_caches() -> None:
    """
    Clear cached clients.

    This is primarily useful in tests that replace environment settings.
    """
    get_chat_model.cache_clear()
    get_embedding_model.cache_clear()
