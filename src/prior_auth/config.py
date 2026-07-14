
"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"

# Load local development variables when .env exists.
load_dotenv(ENV_FILE)


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the prior-authorization service."""

    project_root: Path
    policies_dir: Path
    faiss_index_dir: Path

    openai_api_key: str | None
    openai_model: str
    embedding_model: str

    max_retrieval_attempts: int
    max_critique_attempts: int
    retrieval_top_k: int

    app_name: str
    app_version: str
    environment: str

    def require_openai_api_key(self) -> str:
        """Return the OpenAI key or raise a clear configuration error."""
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. "
                f"Add it to {ENV_FILE} or export it in your shell."
            )

        return self.openai_api_key


def _read_positive_int(name: str, default: int) -> int:
    """Read a positive integer environment variable."""
    raw_value = os.getenv(name, str(default))

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"{name} must be an integer; received {raw_value!r}."
        ) from exc

    if value < 1:
        raise RuntimeError(
            f"{name} must be at least 1; received {value}."
        )

    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings(
        project_root=PROJECT_ROOT,
        policies_dir=PROJECT_ROOT / "policies",
        faiss_index_dir=PROJECT_ROOT / "faiss_index",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv(
            "OPENAI_MODEL",
            "gpt-4o-mini",
        ),
        embedding_model=os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        ),
        max_retrieval_attempts=_read_positive_int(
            "MAX_RETRIEVAL_ATTEMPTS",
            2,
        ),
        max_critique_attempts=_read_positive_int(
            "MAX_CRITIQUE_ATTEMPTS",
            2,
        ),
        retrieval_top_k=_read_positive_int(
            "RETRIEVAL_TOP_K",
            5,
        ),
        app_name=os.getenv(
            "APP_NAME",
            "Agentic Prior-Authorization Copilot",
        ),
        app_version=os.getenv(
            "APP_VERSION",
            "0.1.0",
        ),
        environment=os.getenv(
            "APP_ENVIRONMENT",
            "development",
        ),
    )
