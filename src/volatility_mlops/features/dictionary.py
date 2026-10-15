"""Generate the feature data dictionary from the contract (build-plan Day 9).

Writes a Markdown table of every feature (name, group, window, definition) plus
the target, so the feature set is documented in one tracked place.

Run as a module::

    python -m volatility_mlops.features.dictionary
"""

from __future__ import annotations

from pathlib import Path

from volatility_mlops.features.contract import CONTRACT_PATH, load_feature_contract

# repo_root/src/volatility_mlops/features/dictionary.py -> parents[3] == repo root
OUTPUT_PATH = Path(__file__).resolve().parents[3] / "assets" / "feature_dictionary.md"


def render_dictionary() -> str:
    """Render the feature data dictionary as Markdown."""
    contract = load_feature_contract()
    lines = [
        "# Feature data dictionary",
        "",
        f"Generated from `{CONTRACT_PATH.name}` (feature_version: **{contract.version}**).",
        "",
        "| Feature | Group | Window (days) | Optional | Definition |",
        "|---|---|---|---|---|",
    ]
    for f in contract.features:
        window = "" if f.window is None else str(f.window)
        optional = "yes" if f.optional else ""
        lines.append(f"| `{f.name}` | {f.group} | {window} | {optional} | {f.definition} |")
    lines += [
        "",
        "## Target",
        "",
        "| Column | Definition |",
        "|---|---|",
        "| `target_fwd_rv_21` | annualized realized volatility of the next 21 trading days "
        "(level; modelled in logs) |",
        "| `target_available_on` | date the label first becomes observable (t+21) |",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    """Write the data dictionary to ``assets/feature_dictionary.md``."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render_dictionary())
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
