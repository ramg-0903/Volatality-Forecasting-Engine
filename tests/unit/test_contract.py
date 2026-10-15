"""Unit tests for the feature contract loader."""

from volatility_mlops.features.contract import load_feature_contract


def test_contract_loads_with_a_version_and_unique_names():
    contract = load_feature_contract()

    assert contract.version  # non-empty version string
    names = contract.feature_names()
    assert len(names) == len(set(names)), "feature names must be unique"
    assert len(names) >= 20  # the full section 5.2 catalog


def test_contract_contains_core_features():
    names = load_feature_contract().feature_names()
    for expected in ("log_rv_21", "log_ewma_vol", "vrp_21", "term_spread", "entity_mean_log_rv"):
        assert expected in names


def test_optional_features_can_be_excluded():
    contract = load_feature_contract()
    with_optional = contract.feature_names(include_optional=True)
    without_optional = contract.feature_names(include_optional=False)

    assert "dow" in with_optional
    assert "dow" not in without_optional
    assert len(without_optional) < len(with_optional)
