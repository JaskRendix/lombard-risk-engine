import pandas as pd

from .config import CONCENTRATION_PENALTY, CONCENTRATION_THRESHOLD


def max_single_name_weight(weights: pd.Series) -> float:
    return float(weights.max())


def concentration_addon(weights: pd.Series) -> float:
    max_w = float(weights.max())

    if max_w <= CONCENTRATION_THRESHOLD:
        return 0.0

    excess = max_w - CONCENTRATION_THRESHOLD

    # penalty applied per 10% above threshold
    return CONCENTRATION_PENALTY * (excess / 0.10)
