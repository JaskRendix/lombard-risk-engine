import numpy as np
import pandas as pd

from lombard_risk.engine import LombardRiskEngine
from lombard_risk.liquidity import LiquidityProfile

if __name__ == "__main__":
    # demo: random returns
    dates = pd.date_range("2020-01-01", periods=252)
    returns_df = pd.DataFrame(
        np.random.normal(0, 0.01, size=(252, 3)),
        index=dates,
        columns=["A", "B", "C"],
    )
    weights = pd.Series([0.5, 0.3, 0.2], index=returns_df.columns)

    liq_profiles = {
        "A": LiquidityProfile(isin="A", adv=1_000_000, position_value=300_000),
        "B": LiquidityProfile(isin="B", adv=500_000, position_value=400_000),
        "C": LiquidityProfile(isin="C", adv=200_000, position_value=300_000),
    }

    engine = LombardRiskEngine()
    res = engine.evaluate(
        portfolio_value=1_000_000,
        loan_amount=600_000,
        returns_df=returns_df,
        weights=weights,
        base_haircut=0.20,
        liq_profiles=liq_profiles,
    )

    print(res)
