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
