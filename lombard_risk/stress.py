import pandas as pd


def apply_equity_shock(returns: pd.Series, shock: float = -0.3) -> pd.Series:
    # one-off crash applied at end
    shocked = returns.copy()
    shocked.iloc[-1] += shock
    return shocked


def summarize_stress(base: pd.Series, scenarios: dict[str, pd.Series]) -> pd.DataFrame:
    def cum_ret(r: pd.Series) -> float:
        return (1 + r).prod() - 1

    out = {}
    base_cr = cum_ret(base)
    for name, s in scenarios.items():
        out[name] = {
            "base_cum_return": base_cr,
            "stress_cum_return": cum_ret(s),
        }
    return pd.DataFrame(out).T
