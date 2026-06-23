from dataclasses import dataclass

import pandas as pd

from .concentration import concentration_addon
from .config import ALPHA, MARGIN_CALL_BUFFER
from .haircut import HaircutComponents, vol_addon_from_var
from .liquidity import LiquidityProfile, liquidity_addon
from .ltv import LTVProfile
from .market_models import (
    cornish_fisher_var,
    ewma_es,
    expected_shortfall_from_returns,
    gaussian_es,
    portfolio_returns,
    var_from_returns,
)


@dataclass
class LombardRiskResult:
    haircut_pct: float
    max_ltv_pct: float
    current_ltv_pct: float
    margin_call_threshold_pct: float
    margin_call_triggered: bool
    stress_ltv_pct: float
    post_haircut_value: float
    breach_distance_pct: float
    var_used: float | None = None
    es_hist: float | None = None
    es_gaussian: float | None = None
    es_ewma: float | None = None


class LombardRiskEngine:
    def __init__(
        self, alpha: float = ALPHA, margin_call_buffer: float = MARGIN_CALL_BUFFER
    ):
        self.alpha = alpha
        self.margin_call_buffer = margin_call_buffer

    def evaluate(
        self,
        portfolio_value: float,
        loan_amount: float,
        returns_df: pd.DataFrame,
        weights: pd.Series,
        base_haircut: float,
        liq_profiles: dict[str, LiquidityProfile],
        use_cornish_fisher: bool = False,
    ) -> LombardRiskResult:

        port_ret = portfolio_returns(returns_df, weights)

        if use_cornish_fisher:
            var = cornish_fisher_var(port_ret, self.alpha)
        else:
            var = var_from_returns(port_ret, self.alpha)

        es_hist = expected_shortfall_from_returns(port_ret, self.alpha)
        es_gauss = gaussian_es(port_ret, self.alpha)
        es_ewma = ewma_es(port_ret, self.alpha)

        vol_addon = vol_addon_from_var(abs(var))
        liq_addon = max(liquidity_addon(lp) for lp in liq_profiles.values())
        conc_addon = concentration_addon(weights)

        hc = HaircutComponents(
            base=base_haircut,
            vol_addon=vol_addon,
            liq_addon=liq_addon,
            conc_addon=conc_addon,
        )

        ltv = LTVProfile(
            portfolio_value=portfolio_value,
            loan_amount=loan_amount,
            haircut=hc.total,
        )

        margin_call_threshold = ltv.max_ltv + self.margin_call_buffer

        return LombardRiskResult(
            haircut_pct=hc.total * 100,
            max_ltv_pct=ltv.max_ltv * 100,
            current_ltv_pct=ltv.current_ltv * 100,
            margin_call_threshold_pct=margin_call_threshold * 100,
            margin_call_triggered=ltv.margin_call_triggered(margin_call_threshold),
            stress_ltv_pct=ltv.stress_ltv * 100,
            post_haircut_value=ltv.post_haircut_value,
            breach_distance_pct=ltv.breach_distance(margin_call_threshold) * 100,
            var_used=var,
            es_hist=es_hist,
            es_gaussian=es_gauss,
            es_ewma=es_ewma,
        )
