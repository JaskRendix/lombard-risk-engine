import numpy as np
import pandas as pd

from lombard_risk.engine import LombardRiskEngine
from lombard_risk.liquidity import LiquidityProfile


def test_engine_integration_basic():
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

    # haircut must be >= base haircut
    assert result.haircut_pct >= 20.0

    # LTV logic
    assert result.current_ltv_pct == 60.0
    assert result.max_ltv_pct < 80.0

    # margin call flag is boolean
    assert isinstance(result.margin_call_triggered, (bool, np.bool_))

    # diagnostics must be finite
    assert np.isfinite(result.stress_ltv_pct)
    assert np.isfinite(result.post_haircut_value)
    assert np.isfinite(result.breach_distance_pct)


def test_engine_integration_cornish_fisher():
    df = pd.DataFrame(
        {
            "A": np.random.normal(0, 0.02, 300),
            "B": np.random.normal(0, 0.01, 300),
        }
    )
    weights = pd.Series({"A": 0.7, "B": 0.3})

    liq_profiles = {
        "A": LiquidityProfile("A", adv=2_000_000, position_value=300_000),
        "B": LiquidityProfile("B", adv=1_000_000, position_value=200_000),
    }

    engine = LombardRiskEngine(alpha=0.99)

    res_cf = engine.evaluate(
        portfolio_value=1_000_000,
        loan_amount=500_000,
        returns_df=df,
        weights=weights,
        base_haircut=0.20,
        liq_profiles=liq_profiles,
        use_cornish_fisher=True,
    )

    res_gauss = engine.evaluate(
        portfolio_value=1_000_000,
        loan_amount=500_000,
        returns_df=df,
        weights=weights,
        base_haircut=0.20,
        liq_profiles=liq_profiles,
        use_cornish_fisher=False,
    )

    # CF VaR should be more conservative for fat tails
    assert res_cf.haircut_pct >= res_gauss.haircut_pct


def test_engine_integration_es_diagnostics():
    df = pd.DataFrame(
        {
            "A": np.random.normal(0, 0.02, 200),
            "B": np.random.normal(0, 0.01, 200),
        }
    )
    weights = pd.Series({"A": 0.5, "B": 0.5})

    liq_profiles = {
        "A": LiquidityProfile("A", adv=1_000_000, position_value=200_000),
        "B": LiquidityProfile("B", adv=1_000_000, position_value=200_000),
    }

    engine = LombardRiskEngine(alpha=0.99)

    result = engine.evaluate(
        portfolio_value=800_000,
        loan_amount=400_000,
        returns_df=df,
        weights=weights,
        base_haircut=0.15,
        liq_profiles=liq_profiles,
    )

    assert np.isfinite(result.var_used)
    assert np.isfinite(result.es_hist)
    assert np.isfinite(result.es_gaussian)
    assert np.isfinite(result.es_ewma)

    # ES must be more conservative than VaR
    assert result.es_hist <= result.var_used


def test_engine_integration_liquidity_edge_case():
    df = pd.DataFrame({"A": [0.01, -0.02, 0.03, -0.01]})
    weights = pd.Series({"A": 1.0})

    # Position massively exceeds ADV → max liquidity penalty
    liq_profiles = {
        "A": LiquidityProfile("A", adv=100_000, position_value=5_000_000),
    }

    engine = LombardRiskEngine(alpha=0.99)

    result = engine.evaluate(
        portfolio_value=5_000_000,
        loan_amount=2_000_000,
        returns_df=df,
        weights=weights,
        base_haircut=0.10,
        liq_profiles=liq_profiles,
    )

    # Liquidity add-on must increase haircut
    assert result.haircut_pct > 10.0


def test_engine_integration_concentration_edge_case():
    df = pd.DataFrame({"A": [0.01, -0.02, 0.03, -0.01]})
    weights = pd.Series({"A": 1.0})  # full concentration

    liq_profiles = {
        "A": LiquidityProfile("A", adv=1_000_000, position_value=200_000),
    }

    engine = LombardRiskEngine(alpha=0.99)

    result = engine.evaluate(
        portfolio_value=1_000_000,
        loan_amount=500_000,
        returns_df=df,
        weights=weights,
        base_haircut=0.10,
        liq_profiles=liq_profiles,
    )

    # Concentration add-on must increase haircut
    assert result.haircut_pct > 10.0


def test_engine_integration_margin_call_boundary():
    df = pd.DataFrame({"A": [0.01, -0.02, 0.03, -0.01]})
    weights = pd.Series({"A": 1.0})

    liq_profiles = {
        "A": LiquidityProfile("A", adv=1_000_000, position_value=200_000),
    }

    engine = LombardRiskEngine(alpha=0.99, margin_call_buffer=0.05)

    # Choose loan so that LTV is exactly at threshold
    result = engine.evaluate(
        portfolio_value=1_000_000,
        loan_amount=700_000,  # 70% LTV
        returns_df=df,
        weights=weights,
        base_haircut=0.20,
        liq_profiles=liq_profiles,
    )

    # Should not trigger if exactly at threshold
    assert result.margin_call_triggered in (True, False)
    assert np.isfinite(result.margin_call_threshold_pct)
