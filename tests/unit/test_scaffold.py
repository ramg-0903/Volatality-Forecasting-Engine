"""Day 1 smoke tests: the package is importable and the scaffold is intact.

These are placeholders that keep `make test` green before any component exists.
Delete them once real tests land in week 1.
"""

from pathlib import Path

import volatility_mlops

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_package_imports():
    assert volatility_mlops.__version__ == "0.1.0"


def test_expected_subpackages_exist():
    expected = {
        "ingestion",
        "quality",
        "features",
        "models",
        "simulation",
        "monitoring",
        "api",
        "flows",
    }
    pkg_dir = Path(volatility_mlops.__file__).parent
    actual = {p.name for p in pkg_dir.iterdir() if (p / "__init__.py").exists()}
    assert expected <= actual


def test_env_template_is_committed_and_env_is_not_tracked():
    """.env.template ships with placeholders; a real .env must stay untracked."""
    assert (REPO_ROOT / ".env.template").exists()
    gitignore = (REPO_ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in {line.strip() for line in gitignore}
