"""Load the feature contract (``config/feature_contract.yml``).

The contract is the single source of truth for which features exist. This loader
exposes the version and the ordered feature names for the feature build (Day 9)
and the promotion gate's schema-conformance check (Day 16).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# repo_root/src/volatility_mlops/features/contract.py -> parents[3] == repo root
CONTRACT_PATH = Path(__file__).resolve().parents[3] / "config" / "feature_contract.yml"


@dataclass(frozen=True)
class FeatureSpec:
    """One declared feature.

    Attributes:
        name: Column name, matching the ``features`` table column added on Day 9.
        group: Catalog group (e.g. ``har_ladder``).
        definition: Human-readable definition, stating units and window.
        window: Trailing window in trading days, if the feature has one.
        optional: Whether the feature may be dropped without bumping the version.
    """

    name: str
    group: str
    definition: str
    window: int | None = None
    optional: bool = False


@dataclass(frozen=True)
class FeatureContract:
    """The versioned set of declared features."""

    version: str
    features: tuple[FeatureSpec, ...]

    def feature_names(self, include_optional: bool = True) -> list[str]:
        """Return feature names in contract order.

        Args:
            include_optional: Include features marked ``optional``.

        Returns:
            Ordered list of feature column names.
        """
        return [f.name for f in self.features if include_optional or not f.optional]


def load_feature_contract(path: Path = CONTRACT_PATH) -> FeatureContract:
    """Parse ``feature_contract.yml`` into a :class:`FeatureContract`.

    Args:
        path: Path to the contract YAML.

    Returns:
        The parsed contract.
    """
    data = yaml.safe_load(path.read_text())
    features = tuple(
        FeatureSpec(
            name=f["name"],
            group=f["group"],
            definition=f["definition"],
            window=f.get("window"),
            optional=f.get("optional", False),
        )
        for f in data["features"]
    )
    return FeatureContract(version=str(data["feature_version"]), features=features)
