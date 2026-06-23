import numpy as np

from lombard_risk.ltv import LTVProfile


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
