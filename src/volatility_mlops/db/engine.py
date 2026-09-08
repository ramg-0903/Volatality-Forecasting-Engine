"""SQLAlchemy engine for the application database.

Reads ``DATABASE_URL`` from the environment (``.env`` is loaded if present). That
URL points at the ``volatility`` application database, never at the ``mlflow``
metadata database, which MLflow owns.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def get_engine(url: str | None = None) -> Engine:
    """Build a SQLAlchemy engine for the application database.

    Args:
        url: Explicit SQLAlchemy URL. When omitted, ``DATABASE_URL`` from the
            environment is used.

    Returns:
        A future-style SQLAlchemy engine. ``pool_pre_ping`` is enabled so stale
        connections (for example after the Postgres container restarts) are
        transparently recycled.

    Raises:
        RuntimeError: If no URL is given and ``DATABASE_URL`` is unset.
    """
    load_dotenv()
    resolved = url or os.environ.get("DATABASE_URL")
    if not resolved:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.template to .env (make env) and retry."
        )
    return create_engine(resolved, future=True, pool_pre_ping=True)
