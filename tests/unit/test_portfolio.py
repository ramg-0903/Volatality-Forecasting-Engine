"""Unit tests for portfolio risk math (build-plan section 8.3 / Day 18).

Pure linear algebra, no DB or model. These pin down the properties that make the
endpoint correct: diversification lowers risk, the closed form holds, and the risk
contributions decompose the portfolio volatility exactly.
"""

import numpy as np
import pytest

from volatility_mlops.api.portfolio import (
    covariance_from_vols_and_corr,
    load_serving_config,
    portfolio_risk,
)


def test_perfect_correlation_equals_weighted_average():
    # rho = 1 everywhere -> no diversification -> portfolio vol == weighted average.
    vols = np.array([0.20, 0.30])
    weights = np.array([0.5, 0.5])
    corr = np.array([[1.0, 1.0], [1.0, 1.0]])
    r = portfolio_risk(vols, weights, corr)
    assert r["portfolio_volatility"] == pytest.approx(float(weights @ vols))


@pytest.mark.parametrize("rho", [-0.5, 0.0, 0.3, 0.6, 1.0])
def test_two_asset_closed_form(rho):
    # Equal weight, equal vol: sigma_p = sigma * sqrt((1 + rho) / 2).
    sigma = 0.25
    vols = np.array([sigma, sigma])
    weights = np.array([0.5, 0.5])
    corr = np.array([[1.0, rho], [rho, 1.0]])
    r = portfolio_risk(vols, weights, corr)
    assert r["portfolio_volatility"] == pytest.approx(sigma * np.sqrt((1 + rho) / 2))


def test_lower_correlation_lowers_portfolio_vol():
    vols = np.array([0.20, 0.30])
    weights = np.array([0.4, 0.6])
    high = portfolio_risk(vols, weights, np.array([[1.0, 0.9], [0.9, 1.0]]))
    low = portfolio_risk(vols, weights, np.array([[1.0, 0.1], [0.1, 1.0]]))
    assert low["portfolio_volatility"] < high["portfolio_volatility"]


def test_portfolio_vol_never_above_weighted_average():
    vols = np.array([0.20, 0.30, 0.25])
    weights = np.array([0.5, 0.3, 0.2])
    corr = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]])
    r = portfolio_risk(vols, weights, corr)
    assert r["portfolio_volatility"] <= r["weighted_avg_volatility"] + 1e-12


def test_risk_contributions_sum_to_portfolio_vol():
    vols = np.array([0.20, 0.30, 0.25])
    weights = np.array([0.5, 0.3, 0.2])
    corr = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.4], [0.2, 0.4, 1.0]])
    r = portfolio_risk(vols, weights, corr)
    assert r["risk_contributions"].sum() == pytest.approx(r["portfolio_volatility"])


def test_single_holding_reduces_to_its_own_vol():
    r = portfolio_risk(np.array([0.27]), np.array([1.0]), np.array([[1.0]]))
    assert r["portfolio_volatility"] == pytest.approx(0.27)
    assert r["risk_contributions"][0] == pytest.approx(0.27)


def test_covariance_from_vols_and_corr():
    vols = np.array([0.20, 0.30])
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    cov = covariance_from_vols_and_corr(vols, corr)
    assert cov[0, 0] == pytest.approx(0.04)  # 0.20^2
    assert cov[1, 1] == pytest.approx(0.09)  # 0.30^2
    assert cov[0, 1] == pytest.approx(0.5 * 0.20 * 0.30)


def test_serving_config_loads():
    cfg = load_serving_config()
    assert cfg.correlation_window_days == 126
    assert cfg.min_observations == 60
