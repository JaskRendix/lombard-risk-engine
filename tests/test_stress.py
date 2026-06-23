import pandas as pd
import pytest

from lombard_risk.stress import apply_equity_shock, summarize_stress


def test_apply_equity_shock_basic():
    s = pd.Series([0.01, 0.02, 0.03])
    out = apply_equity_shock(s, shock=-0.3)
    assert out.iloc[-1] == 0.03 - 0.3
    assert out.iloc[:-1].equals(s.iloc[:-1])


@pytest.mark.parametrize("shock", [-0.1, -0.5, 0.2])
def test_apply_equity_shock_parametrized(shock):
    s = pd.Series([0.0, 0.0, 0.0])
    out = apply_equity_shock(s, shock=shock)
    assert out.iloc[-1] == shock


def test_apply_equity_shock_no_mutation():
    s = pd.Series([0.01, 0.02, 0.03])
    s_copy = s.copy()
    _ = apply_equity_shock(s, shock=-0.3)
    assert s.equals(s_copy)


def test_summarize_stress_basic():
    base = pd.Series([0.01, -0.02, 0.03])
    scen = {
        "crash": pd.Series([0.01, -0.02, -0.30]),
        "mild": pd.Series([0.01, -0.02, 0.00]),
    }

    df = summarize_stress(base, scen)

    assert set(df.index) == {"crash", "mild"}
    assert "base_cum_return" in df.columns
    assert "stress_cum_return" in df.columns


def test_summarize_stress_empty():
    base = pd.Series([0.01, 0.02])
    df = summarize_stress(base, {})
    assert df.empty
