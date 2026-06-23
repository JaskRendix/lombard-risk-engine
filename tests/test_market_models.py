import numpy as np
import pandas as pd
import pytest

from lombard_risk.config import ALPHA, EWMA_LAMBDA
from lombard_risk.market_models import (
    cornish_fisher_var,
    ewma_es,
    ewma_vol,
    expected_shortfall_from_returns,
    gaussian_es,
    portfolio_returns,
    sample_kurtosis,
    sample_skewness,
    scale_var,
    scale_volatility,
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


def test_expected_shortfall_basic():
    r = pd.Series([-0.10, -0.05, 0.01, 0.02])
    es = expected_shortfall_from_returns(r, 0.95)
    assert es <= -0.05  # ES must be <= worst 5% losses
    assert np.isfinite(es)


def test_expected_shortfall_zero():
    r = pd.Series([0.0, 0.0, 0.0])
    assert expected_shortfall_from_returns(r) == 0.0


def test_expected_shortfall_monotonicity():
    mild = pd.Series([-0.01, -0.02, 0.00])
    severe = pd.Series([-0.10, -0.20, -0.30])
    assert expected_shortfall_from_returns(severe) < expected_shortfall_from_returns(
        mild
    )


def test_gaussian_es_negative_for_normal():
    np.random.seed(0)
    r = pd.Series(np.random.normal(0, 0.01, size=5000))
    es = gaussian_es(r, 0.99)
    assert es < 0


def test_gaussian_es_more_conservative_than_var():
    np.random.seed(0)
    r = pd.Series(np.random.normal(0, 0.01, size=5000))
    var = var_from_returns(r, 0.99)
    es = gaussian_es(r, 0.99)
    assert es < var  # ES is always more conservative


def test_ewma_es_basic():
    r = pd.Series([-0.10, -0.02, 0.01, 0.03])
    es = ewma_es(r, 0.95)
    assert np.isfinite(es)


def test_ewma_es_zero():
    r = pd.Series([0.0, 0.0, 0.0])
    assert ewma_es(r) == 0.0


def test_ewma_es_monotonicity():
    mild = pd.Series([-0.01, -0.02, 0.00])
    severe = pd.Series([-0.10, -0.20, -0.30])
    assert ewma_es(severe) < ewma_es(mild)


def test_scale_volatility_basic():
    assert scale_volatility(0.10, 20, 10) == pytest.approx(0.10 * np.sqrt(2))


def test_scale_volatility_zero():
    assert scale_volatility(0.0, 20) == 0.0


def test_scale_volatility_monotonicity():
    assert scale_volatility(0.10, 20) > scale_volatility(0.10, 10)


def test_sample_skewness_zero_for_symmetric():
    r = pd.Series([-1, 0, 1])
    assert abs(sample_skewness(r)) < 1e-6


def test_sample_kurtosis_zero_for_normal_like():
    np.random.seed(0)
    r = pd.Series(np.random.normal(size=5000))
    assert abs(sample_kurtosis(r)) < 0.2  # approximate


def test_sample_skewness_sign():
    r = pd.Series([-3, -2, -1, 0, 10])
    assert sample_skewness(r) > 0  # right tail


def test_sample_kurtosis_heavy_tails():
    r = pd.Series(np.random.standard_t(df=3, size=5000))
    assert sample_kurtosis(r) > 0


def test_scale_var_identity():
    # 10‑day horizon = base horizon → no scaling
    assert scale_var(0.05, 10) == 0.05


def test_scale_var_longer_horizon():
    # 20‑day horizon → sqrt(2) scaling
    out = scale_var(0.10, 20)
    expected = 0.10 * np.sqrt(2)
    assert np.isclose(out, expected)


def test_scale_var_shorter_horizon():
    # 5‑day horizon → sqrt(0.5) scaling
    out = scale_var(0.10, 5)
    expected = 0.10 * np.sqrt(0.5)
    assert np.isclose(out, expected)


def test_scale_var_zero():
    assert scale_var(0.0, 20) == 0.0


def test_scale_var_monotonicity():
    v = 0.10
    assert scale_var(v, 20) > scale_var(v, 10)
    assert scale_var(v, 10) > scale_var(v, 5)
