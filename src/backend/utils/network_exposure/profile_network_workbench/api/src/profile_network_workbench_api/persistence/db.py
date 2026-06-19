from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - lean CLI environments
    def load_dotenv(*args: object, **kwargs: object) -> bool:
        return False


def lab_root() -> Path:
    return Path(__file__).resolve().parents[3]


def database_url(explicit_url: str | None = None) -> str | None:
    load_dotenv(lab_root() / ".env")
    value = explicit_url or os.getenv("DATABASE_URL")
    if value is None or not value.strip():
        return None
    return value.strip()


def database_url_required(explicit_url: str | None = None) -> str:
    value = database_url(explicit_url)
    if value is None:
        raise RuntimeError("DATABASE_URL is not configured.")
    return value


@contextmanager
def connect(explicit_url: str | None = None) -> Iterator[object]:
    try:
        import psycopg
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("psycopg is required for Postgres persistence. Install psycopg[binary].") from exc

    with psycopg.connect(database_url_required(explicit_url)) as conn:
        yield conn
