import pytest

from lombard_risk.liquidity import LiquidityProfile, liquidity_addon


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
    lp = LiquidityProfile("X", adv=adv, position_value=position_value)
    assert liquidity_addon(lp) == expected
