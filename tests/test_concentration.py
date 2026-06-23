import numpy as np
import pandas as pd
import pytest

from lombard_risk.concentration import concentration_addon, max_single_name_weight
from lombard_risk.config import CONCENTRATION_PENALTY, CONCENTRATION_THRESHOLD


@pytest.mark.parametrize(
    "weights,expected",
    [
        (pd.Series([0.2, 0.3, 0.1]), 0.3),
        (pd.Series([0.0, 0.0, 0.0]), 0.0),
        (pd.Series([1.0]), 1.0),
    ],
)
def test_max_single_name_weight(weights, expected):
    assert max_single_name_weight(weights) == expected


@pytest.mark.parametrize(
    "weights,expected",
    [
        # No concentration: max weight == threshold
        (pd.Series([CONCENTRATION_THRESHOLD] * 5), 0.0),
        # Slightly above threshold
        (
            pd.Series([CONCENTRATION_THRESHOLD + 0.05, 0.1, 0.1]),
            CONCENTRATION_PENALTY * (0.05 / 0.10),
        ),
        # Well above threshold
        (
            pd.Series([CONCENTRATION_THRESHOLD + 0.20, 0.1, 0.1]),
            CONCENTRATION_PENALTY * (0.20 / 0.10),
        ),
    ],
)
def test_concentration_addon_parametrized(weights, expected):
    out = concentration_addon(weights)
    assert np.isclose(out, expected)


def test_concentration_addon_zero_weights():
    weights = pd.Series([0.0, 0.0, 0.0])
    assert concentration_addon(weights) == 0.0


def test_concentration_addon_single_asset():
    weights = pd.Series([1.0])
    expected = CONCENTRATION_PENALTY * ((1.0 - CONCENTRATION_THRESHOLD) / 0.10)
    assert np.isclose(concentration_addon(weights), expected)


def test_concentration_addon_monotonicity():
    """Higher max weight must produce higher penalty."""
    w1 = pd.Series([0.40, 0.30, 0.30])
    w2 = pd.Series([0.60, 0.20, 0.20])

    assert concentration_addon(w2) > concentration_addon(w1)
