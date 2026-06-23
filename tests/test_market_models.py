import numpy as np
import pandas as pd
import pytest

from lombard_risk.config import ALPHA, EWMA_LAMBDA
from lombard_risk.market_models import (
    cornish_fisher_var,
    ewma_vol,
    portfolio_returns,
    var_from_returns,
)


def test_ewma_vol_basic():
    returns = pd.Series([0.01, -0.02, 0.03])
    lam = EWMA_LAMBDA

    weights = np.array([lam**i for i in range(len(returns))])[::-1]
    weights /= weights.sum()

    expected = np.sqrt((weights * returns**2).sum())
    assert np.isclose(ewma_vol(returns), expected)


@pytest.mark.parametrize(
    "series",
    [
        pd.Series([0.0, 0.0, 0.0]),
        pd.Series([0.01, 0.01, 0.01]),
        pd.Series([-0.02, 0.03, -0.01]),
    ],
)
def test_ewma_vol_parametrized(series):
    vol = ewma_vol(series)
    assert vol >= 0.0
    assert np.isfinite(vol)


def test_ewma_vol_zero_returns():
    returns = pd.Series([0.0, 0.0, 0.0, 0.0])
    assert ewma_vol(returns) == 0.0


def test_ewma_vol_monotonicity():
    small = pd.Series([0.01, 0.01, 0.01])
    large = pd.Series([0.05, 0.05, 0.05])
    assert ewma_vol(large) > ewma_vol(small)


def test_portfolio_returns_basic():
    df = pd.DataFrame({"A": [0.01, -0.02, 0.03], "B": [0.02, 0.01, -0.01]})
    weights = pd.Series({"A": 0.6, "B": 0.4})

    out = portfolio_returns(df, weights)

    expected = pd.Series(
        [
            0.6 * 0.01 + 0.4 * 0.02,
            0.6 * (-0.02) + 0.4 * 0.01,
            0.6 * 0.03 + 0.4 * (-0.01),
        ]
    )

    assert np.allclose(out.values, expected.values)


def test_portfolio_returns_single_asset():
    df = pd.DataFrame({"A": [0.01, 0.02, -0.01]})
    weights = pd.Series({"A": 1.0})
    out = portfolio_returns(df, weights)
    assert out.equals(df["A"])


def test_portfolio_returns_zero_weights():
    df = pd.DataFrame({"A": [0.01, 0.02], "B": [-0.01, 0.03]})
    weights = pd.Series({"A": 0.0, "B": 0.0})
    out = portfolio_returns(df, weights)
    assert np.allclose(out.values, [0.0, 0.0])


@pytest.mark.parametrize(
    "returns,alpha,expected",
    [
        (pd.Series([0.01, -0.02, 0.03, -0.05]), 0.99, -0.0491),
        (pd.Series([0.01, 0.02, 0.03]), 0.95, 0.011),
    ],
)
def test_var_from_returns_parametrized(returns, alpha, expected):
    out = var_from_returns(returns, alpha)
    assert np.isclose(out, expected, atol=1e-4)


def test_var_from_returns_zero():
    returns = pd.Series([0.0, 0.0, 0.0])
    assert var_from_returns(returns) == 0.0


def test_var_from_returns_monotonicity():
    mild = pd.Series([-0.01, -0.02, 0.00])
    severe = pd.Series([-0.10, -0.20, -0.30])
    assert var_from_returns(severe) < var_from_returns(mild)


def test_var_from_returns_default_alpha():
    returns = pd.Series([-0.01, -0.02, -0.03, 0.01])
    expected = np.quantile(returns, 1 - ALPHA)
    assert var_from_returns(returns) == expected


def test_cornish_fisher_var_negative():
    np.random.seed(0)
    r = np.random.normal(0, 0.01, size=5000)
    var_cf = cornish_fisher_var(r, 0.99)
    assert var_cf < 0


def test_cornish_fisher_equals_gaussian_for_normal():
    np.random.seed(0)
    r = np.random.normal(0, 0.01, size=20000)

    var_gauss = var_from_returns(pd.Series(r), 0.99)
    var_cf = cornish_fisher_var(r, 0.99)

    # CF should be extremely close to Gaussian
    assert np.isclose(var_cf, var_gauss, atol=1e-3)


def test_cornish_fisher_more_conservative_for_fat_tails():
    np.random.seed(0)

    # Student-t with df=3 has heavy tails
    t = np.random.standard_t(df=3, size=20000) * 0.01

    var_gauss = var_from_returns(pd.Series(t), 0.99)
    var_cf = cornish_fisher_var(t, 0.99)

    # CF VaR should be more negative (larger loss)
    assert var_cf < var_gauss


def test_cornish_fisher_small_sample_stability():
    r = np.array([0.01, -0.02, 0.03])
    var_cf = cornish_fisher_var(r, 0.95)
    assert np.isfinite(var_cf)
