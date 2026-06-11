import numpy as np
import pandas as pd

from .config import ALPHA, EWMA_LAMBDA


def ewma_vol(returns: pd.Series) -> float:
    lam = EWMA_LAMBDA
    weights = np.array([lam**i for i in range(len(returns))])[::-1]
    weights /= weights.sum()
    return np.sqrt((weights * returns**2).sum())


def portfolio_returns(returns_df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    return returns_df.dot(weights)


def var_from_returns(returns: pd.Series, alpha: float = ALPHA) -> float:
    return np.quantile(returns, 1 - alpha)
