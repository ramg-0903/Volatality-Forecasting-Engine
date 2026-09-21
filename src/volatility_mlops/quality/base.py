"""Result types for the data quality gate.

A check is a pure function over a normalized OHLCV frame that returns a
:class:`CheckResult`. Severity is attached by the runner from ``config/quality.yml``;
checks themselves are severity-agnostic so the same check can gate in one context
and merely warn in another.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Severity(StrEnum):
    """How a failed check is treated by the runner."""

    ERROR = "error"  # failure aborts the pipeline (raises)
    WARN = "warn"  # failure is recorded only


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one data quality check.

    Attributes:
        check_name: Stable identifier, matching a key in ``config/quality.yml``.
        target_table: The table the check ran against (persisted for auditing).
        passed: Whether the check passed.
        observed_value: The measured quantity (e.g. count of violations); units
            depend on the check. ``None`` when not applicable.
        threshold: The threshold compared against, when the check has one.
        detail: Human-readable context for logs and alerts (not persisted).
        severity: Attached by the runner; defaults to ``ERROR``.
    """

    check_name: str
    target_table: str
    passed: bool
    observed_value: float | None = None
    threshold: float | None = None
    detail: str = ""
    severity: Severity = Severity.ERROR
