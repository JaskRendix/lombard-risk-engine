from dataclasses import dataclass

from .config import LIQUIDITY_DISCOUNTS, LIQUIDITY_THRESHOLDS


@dataclass
class LiquidityProfile:
    isin: str
    adv: float
    position_value: float


def liquidity_addon(lp: LiquidityProfile) -> float:
    ratio = lp.position_value / lp.adv

    if ratio < LIQUIDITY_THRESHOLDS["low"]:
        return LIQUIDITY_DISCOUNTS["low"]

    elif ratio < LIQUIDITY_THRESHOLDS["medium"]:
        return LIQUIDITY_DISCOUNTS["medium"]

    elif ratio < LIQUIDITY_THRESHOLDS["high"]:
        return LIQUIDITY_DISCOUNTS["high"]

    else:
        return LIQUIDITY_DISCOUNTS["extreme"]
