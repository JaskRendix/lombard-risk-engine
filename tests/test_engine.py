import numpy as np
import pandas as pd
import pytest

from lombard_risk.concentration import concentration_addon
from lombard_risk.engine import LombardRiskEngine
from lombard_risk.haircut import HaircutComponents, vol_addon_from_var
from lombard_risk.liquidity import LiquidityProfile, liquidity_addon
from lombard_risk.ltv import LTVProfile
from lombard_risk.market_models import portfolio_returns, var_from_returns


def test_portfolio_returns_basic():
    df = pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.03],
            "B": [0.02, 0.01, -0.01],
        }
    )
    weights = pd.Series({"A": 0.6, "B": 0.4})

    out = portfolio_returns(df, weights)

    expected = pd.Series(
        [
            0.6 * 0.01 + 0.4 * 0.02,
            0.6 * -0.02 + 0.4 * 0.01,
            0.6 * 0.03 + 0.4 * -0.01,
        ]
    )

    assert np.allclose(out.values, expected.values)


@pytest.mark.parametrize(
    "returns,alpha,expected",
    [
        (pd.Series([0.01, -0.02, 0.03, -0.05]), 0.99, -0.0491),
        (pd.Series([0.01, 0.02, 0.03]), 0.95, 0.011),
    ],
)
def test_var_from_returns(returns, alpha, expected):
    out = var_from_returns(returns, alpha)
    assert np.isclose(out, expected, atol=1e-4)


@pytest.mark.parametrize(
    "position_value,adv,expected",
    [
        (50_000, 1_000_000, 0.00),
        (300_000, 1_000_000, 0.02),
        (800_000, 1_000_000, 0.05),
        (2_000_000, 1_000_000, 0.10),
    ],
)
def test_liquidity_addon(position_value, adv, expected):
    lp = LiquidityProfile(isin="X", adv=adv, position_value=position_value)
    assert liquidity_addon(lp) == expected


@pytest.mark.parametrize(
    "weights,expected",
    [
        (pd.Series([0.2, 0.2, 0.2, 0.2, 0.2]), 0.0),
        (pd.Series([0.3, 0.2, 0.2, 0.2, 0.1]), 0.05),
        (pd.Series([0.5, 0.2, 0.2, 0.1, 0.0]), 0.15),
    ],
)
def test_concentration_addon(weights, expected):
    out = concentration_addon(weights)
    assert np.isclose(out, expected)


def test_haircut_components_total():
    hc = HaircutComponents(
        base=0.20,
        vol_addon=0.03,
        liq_addon=0.02,
        conc_addon=0.05,
    )
    assert np.isclose(hc.total, 0.30)


def test_vol_addon_from_var():
    # VOL_BUFFER = 0.02 in config
    assert np.isclose(vol_addon_from_var(0.10), 0.12)


def test_ltv_profile_basic():
    ltv = LTVProfile(
        portfolio_value=1_000_000,
        loan_amount=600_000,
        haircut=0.25,
    )

    assert np.isclose(ltv.max_ltv, 0.75)
    assert np.isclose(ltv.current_ltv, 0.60)
    assert not ltv.margin_call_triggered(0.70)
    assert ltv.margin_call_triggered(0.55)


def test_engine_integration_simple():
    df = pd.DataFrame(
        {
            "A": [0.01, -0.02, 0.03, -0.01],
            "B": [0.02, 0.01, -0.01, 0.00],
        }
    )
    weights = pd.Series({"A": 0.5, "B": 0.5})

    liq_profiles = {
        "A": LiquidityProfile("A", adv=1_000_000, position_value=200_000),
        "B": LiquidityProfile("B", adv=500_000, position_value=400_000),
    }

    engine = LombardRiskEngine(alpha=0.99, margin_call_buffer=0.10)

    result = engine.evaluate(
        portfolio_value=1_000_000,
        loan_amount=600_000,
        returns_df=df,
        weights=weights,
        base_haircut=0.20,
        liq_profiles=liq_profiles,
    )

    assert result.haircut_pct > 20.0
    assert result.max_ltv_pct < 80.0
    assert result.current_ltv_pct == 60.0
    assert result.margin_call_threshold_pct > result.max_ltv_pct
    assert isinstance(result.margin_call_triggered, (bool, np.bool_))


def test_zero_volatility_returns():
    df = pd.DataFrame({"A": [0.0, 0.0, 0.0, 0.0]})
    weights = pd.Series({"A": 1.0})

    liq_profiles = {
        "A": LiquidityProfile("A", adv=1_000_000, position_value=10_000),
    }

    engine = LombardRiskEngine()

    result = engine.evaluate(
        portfolio_value=500_000,
        loan_amount=100_000,
        returns_df=df,
        weights=weights,
        base_haircut=0.10,
        liq_profiles=liq_profiles,
    )

    assert result.haircut_pct > 10.0


def test_extreme_concentration():
    df = pd.DataFrame({"A": [0.01, -0.02, 0.03]})
    weights = pd.Series({"A": 1.0})

    liq_profiles = {
        "A": LiquidityProfile("A", adv=100_000, position_value=200_000),
    }

    engine = LombardRiskEngine()

    result = engine.evaluate(
        portfolio_value=1_000_000,
        loan_amount=900_000,
        returns_df=df,
        weights=weights,
        base_haircut=0.10,
        liq_profiles=liq_profiles,
    )

    assert result.haircut_pct > 10.0
    assert isinstance(result.margin_call_triggered, (bool, np.bool_))


def test_negative_returns_extreme_var():
    df = pd.DataFrame({"A": [-0.10, -0.20, -0.30, -0.40]})
    weights = pd.Series({"A": 1.0})

    liq_profiles = {
        "A": LiquidityProfile("A", adv=1_000_000, position_value=100_000),
    }

    engine = LombardRiskEngine()

    result = engine.evaluate(
        portfolio_value=1_000_000,
        loan_amount=500_000,
        returns_df=df,
        weights=weights,
        base_haircut=0.10,
        liq_profiles=liq_profiles,
    )

    assert result.haircut_pct > 20.0


def test_engine_diagnostics_fields():
    df = pd.DataFrame({"A": [0.01, -0.02, 0.03]})
    weights = pd.Series({"A": 1.0})

    liq_profiles = {
        "A": LiquidityProfile("A", adv=1_000_000, position_value=100_000),
    }

    engine = LombardRiskEngine()

    result = engine.evaluate(
        portfolio_value=1_000_000,
        loan_amount=500_000,
        returns_df=df,
        weights=weights,
        base_haircut=0.10,
        liq_profiles=liq_profiles,
    )

    # New diagnostics exist and are numeric
    assert isinstance(result.stress_ltv_pct, float)
    assert isinstance(result.post_haircut_value, float)
    assert isinstance(result.breach_distance_pct, float)

    # Stress LTV must be >= current LTV
    assert result.stress_ltv_pct >= result.current_ltv_pct

    # Post haircut value must be < portfolio value
    assert result.post_haircut_value < 1_000_000
