"""Configuration helpers that avoid adding python-dotenv as a dependency."""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path = ".env.local") -> None:
    """Load simple KEY=VALUE lines into os.environ if they are not already set."""

    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_api_key() -> str:
    load_env_file()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GEMINI_API_KEY or GOOGLE_API_KEY in the environment or .env.local."
        )
    return api_key


def get_embedding_model() -> str:
    load_env_file()
    return os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")


def get_generation_model() -> str:
    load_env_file()
    return os.getenv("GEMINI_GENERATION_MODEL", "gemini-2.5-flash-lite")
