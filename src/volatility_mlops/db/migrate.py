"""Forward-only SQL migration runner.

Applies the numbered ``*.sql`` files in ``db/migrations/`` to the application
database in lexical order, exactly once each, inside one transaction per file.
Re-running is a no-op: already-applied versions are skipped, which is the
idempotency guarantee the day-2 done-when requires.

An applied file's checksum is recorded. If a file changes after being applied,
the runner refuses to continue: migrations are forward-only, so a fix ships as a
new file, never as an edit to an old one.

Run as a module::

    python -m volatility_mlops.db.migrate
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine

from volatility_mlops.db.engine import get_engine

# repo_root/src/volatility_mlops/db/migrate.py -> parents[3] == repo root
MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "db" / "migrations"

_BOOKKEEPING_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     text PRIMARY KEY,
    checksum    text NOT NULL,
    applied_at  timestamptz NOT NULL DEFAULT now()
);
"""


@dataclass(frozen=True)
class Migration:
    """A single migration file discovered on disk."""

    version: str
    path: Path
    checksum: str


def _checksum(data: bytes) -> str:
    """Return the SHA-256 hex digest of the given bytes."""
    return hashlib.sha256(data).hexdigest()


def discover_migrations(directory: Path = MIGRATIONS_DIR) -> list[Migration]:
    """List migration files in lexical (version) order.

    Args:
        directory: Folder holding the numbered ``*.sql`` files.

    Returns:
        Migrations sorted by filename, each with its content checksum.
    """
    files = sorted(p for p in directory.glob("*.sql"))
    return [Migration(p.name, p, _checksum(p.read_bytes())) for p in files]


def _ensure_bookkeeping(engine: Engine) -> None:
    """Create the ``schema_migrations`` tracking table if it does not exist."""
    with engine.begin() as conn:
        conn.exec_driver_sql(_BOOKKEEPING_DDL)


def _applied(engine: Engine) -> dict[str, str]:
    """Return already-applied versions mapped to their recorded checksum."""
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT version, checksum FROM schema_migrations"))
        return {row.version: row.checksum for row in rows}


def apply_migrations(
    engine: Engine | None = None,
    directory: Path = MIGRATIONS_DIR,
) -> list[str]:
    """Apply every not-yet-applied migration, in order.

    Each file runs in its own transaction together with the insert that records
    it, so a failure rolls the file back and leaves it unrecorded.

    Args:
        engine: Target engine. Defaults to :func:`get_engine`.
        directory: Folder holding the migration files.

    Returns:
        The versions applied by this call, in order. Empty if already current.

    Raises:
        RuntimeError: If an already-applied file's checksum no longer matches
            (a forward-only violation).
    """
    engine = engine or get_engine()
    _ensure_bookkeeping(engine)
    already = _applied(engine)

    newly: list[str] = []
    for migration in discover_migrations(directory):
        recorded = already.get(migration.version)
        if recorded is not None:
            if recorded != migration.checksum:
                raise RuntimeError(
                    f"Migration {migration.version} changed after being applied "
                    "(checksum drift). Migrations are forward-only: add a new "
                    "file instead of editing this one."
                )
            continue
        sql = migration.path.read_text()
        with engine.begin() as conn:
            conn.exec_driver_sql(sql)
            conn.execute(
                text(
                    "INSERT INTO schema_migrations (version, checksum) VALUES (:version, :checksum)"
                ),
                {"version": migration.version, "checksum": migration.checksum},
            )
        newly.append(migration.version)
    return newly


def main() -> int:
    """Apply pending migrations and print a short summary. Returns an exit code."""
    newly = apply_migrations()
    if newly:
        print(f"Applied {len(newly)} migration(s):")
        for version in newly:
            print(f"  + {version}")
    else:
        print("Database is up to date; no migrations to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
