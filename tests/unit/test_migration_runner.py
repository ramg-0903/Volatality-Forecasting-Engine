"""Unit tests for the migration runner's pure logic (no database required)."""

from pathlib import Path

from volatility_mlops.db.migrate import discover_migrations


def test_discover_migrations_is_sorted_and_checksummed(tmp_path: Path):
    # Written out of order on purpose; discovery must return them sorted.
    (tmp_path / "0002_second.sql").write_text("SELECT 2;")
    (tmp_path / "0001_first.sql").write_text("SELECT 1;")

    migrations = discover_migrations(tmp_path)

    assert [m.version for m in migrations] == ["0001_first.sql", "0002_second.sql"]
    assert all(len(m.checksum) == 64 for m in migrations)  # sha-256 hex


def test_checksum_changes_with_content(tmp_path: Path):
    path = tmp_path / "0001_first.sql"
    path.write_text("SELECT 1;")
    first = discover_migrations(tmp_path)[0].checksum

    path.write_text("SELECT 2;")
    second = discover_migrations(tmp_path)[0].checksum

    assert first != second


def test_real_migrations_dir_has_the_initial_schema():
    """The committed migrations folder contains the numbered initial schema."""
    versions = [m.version for m in discover_migrations()]
    assert "0001_initial_schema.sql" in versions
