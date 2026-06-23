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
    # left-tail loss quantile
    return np.quantile(returns, 1 - alpha)


def _norm_ppf(p: float) -> float:
    """Inverse CDF of standard normal (Abramowitz–Stegun approximation)."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0, 1)")

    # Coefficients for central region
    a1 = -39.6968302866538
    a2 = 220.946098424521
    a3 = -275.928510446969
    a4 = 138.357751867269
    a5 = -30.6647980661472
    a6 = 2.50662827745924

    b1 = -54.4760987982241
    b2 = 161.585836858041
    b3 = -155.698979859887
    b4 = 66.8013118877197
    b5 = -13.2806815528857

    # Coefficients for tails
    c1 = -0.00778489400243029
    c2 = -0.322396458041136
    c3 = -2.40075827716184
    c4 = -2.54973253934373
    c5 = 4.37466414146497
    c6 = 2.93816398269878

    d1 = 0.00778469570904146
    d2 = 0.32246712907004
    d3 = 2.445134137143
    d4 = 3.75440866190742

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        num = ((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6
        den = (((d1 * q + d2) * q + d3) * q + d4) * q + 1
        return num / den

    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        num = ((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6
        den = (((d1 * q + d2) * q + d3) * q + d4) * q + 1
        return -num / den

    q = p - 0.5
    r = q * q
    num = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
    den = ((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1
    return num / den


def cornish_fisher_var(returns: pd.Series, alpha: float = ALPHA) -> float:
    r = np.asarray(returns)
    mu = r.mean()
    sigma = r.std(ddof=1)

    if sigma == 0:
        return 0.0

    g1 = ((r - mu) ** 3).mean() / sigma**3
    g2 = ((r - mu) ** 4).mean() / sigma**4 - 3  # excess kurtosis

    # left tail: 1 - alpha (e.g. 0.01 for 99% VaR)
    z = _norm_ppf(1 - alpha)

    z_cf = (
        z
        + (1 / 6) * (z**2 - 1) * g1
        + (1 / 24) * (z**3 - 3 * z) * g2
        - (1 / 36) * (2 * z**3 - 5 * z) * (g1**2)
    )

    return mu + z_cf * sigma


def expected_shortfall_from_returns(returns: pd.Series, alpha: float = ALPHA) -> float:
    """Historical Expected Shortfall (left tail)."""
    r = np.asarray(returns)
    var = np.quantile(r, 1 - alpha)
    tail = r[r < var]
    if len(tail) == 0:
        return var  # fallback
    return tail.mean()


def gaussian_es(returns: pd.Series, alpha: float = ALPHA) -> float:
    r = np.asarray(returns)
    mu = r.mean()
    sigma = r.std(ddof=1)
    if sigma == 0:
        return mu

    z = _norm_ppf(alpha)
    pdf = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)

    es = mu - sigma * pdf / (1 - alpha)
    return es


def ewma_es(returns: pd.Series, alpha: float = ALPHA) -> float:
    r = np.asarray(returns)
    lam = EWMA_LAMBDA

    # EWMA weights
    w = np.array([lam**i for i in range(len(r))])[::-1]
    w /= w.sum()

    # Weighted VaR
    sorted_r = np.sort(r)
    sorted_w = w[np.argsort(r)]
    cum_w = np.cumsum(sorted_w)

    idx = np.searchsorted(cum_w, 1 - alpha)
    var = sorted_r[idx]

    # Weighted ES
    tail_mask = sorted_r < var
    if not tail_mask.any():
        return var

    return np.average(sorted_r[tail_mask], weights=sorted_w[tail_mask])


def scale_volatility(vol: float, horizon_days: int, base_horizon: int = 10) -> float:
    return vol * np.sqrt(horizon_days / base_horizon)


def sample_skewness(returns: pd.Series) -> float:
    r = np.asarray(returns)
    mu = r.mean()
    sigma = r.std(ddof=1)
    return ((r - mu) ** 3).mean() / sigma**3 if sigma > 0 else 0.0


def sample_kurtosis(returns: pd.Series) -> float:
    r = np.asarray(returns)
    mu = r.mean()
    sigma = r.std(ddof=1)
    return ((r - mu) ** 4).mean() / sigma**4 - 3 if sigma > 0 else 0.0
