from .config import WWR_LAMBDA


def wwr_multiplier(correlation: float) -> float:
    """
    correlation: borrower–collateral correlation in [-1, 1]
    """
    return 1.0 + WWR_LAMBDA * correlation
