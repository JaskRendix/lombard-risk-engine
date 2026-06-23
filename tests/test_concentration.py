import numpy as np
import pandas as pd
import pytest

from lombard_risk.concentration import (
    concentration_addon,
    max_single_name_weight,
    quadratic_concentration_addon,
)
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
    assert concentration_addon(pd.Series([0.0, 0.0, 0.0])) == 0.0


def test_concentration_addon_single_asset():
    w = pd.Series([1.0])
    expected = CONCENTRATION_PENALTY * ((1.0 - CONCENTRATION_THRESHOLD) / 0.10)
    assert np.isclose(concentration_addon(w), expected)


def test_concentration_addon_monotonicity():
    """Higher max weight must produce higher penalty."""
    w1 = pd.Series([0.40, 0.30, 0.30])
    w2 = pd.Series([0.60, 0.20, 0.20])
    assert concentration_addon(w2) > concentration_addon(w1)


def test_quadratic_concentration_zero_below_threshold():
    w = pd.Series([CONCENTRATION_THRESHOLD - 0.01])
    assert quadratic_concentration_addon(w) == 0.0


def test_quadratic_concentration_monotonicity():
    w1 = pd.Series([CONCENTRATION_THRESHOLD + 0.05])
    w2 = pd.Series([CONCENTRATION_THRESHOLD + 0.15])
    assert quadratic_concentration_addon(w2) > quadratic_concentration_addon(w1)


def test_quadratic_penalty_grows_faster_than_linear():
    """Quadratic penalty accelerates faster than linear as excess increases."""
    w_small = pd.Series([CONCENTRATION_THRESHOLD + 0.10])
    w_large = pd.Series([CONCENTRATION_THRESHOLD + 0.50])

    quad_small = quadratic_concentration_addon(w_small)
    quad_large = quadratic_concentration_addon(w_large)

    lin_small = concentration_addon(w_small)
    lin_large = concentration_addon(w_large)

    # Quadratic growth ratio must exceed linear growth ratio
    assert (quad_large / quad_small) > (lin_large / lin_small)


def test_quadratic_concentration_convexity():
    """Quadratic penalty must be convex: f(2x) > 2 f(x)."""
    x = 0.05
    w1 = pd.Series([CONCENTRATION_THRESHOLD + x])
    w2 = pd.Series([CONCENTRATION_THRESHOLD + 2 * x])

    f1 = quadratic_concentration_addon(w1)
    f2 = quadratic_concentration_addon(w2)

    assert f2 > 2 * f1


def test_quadratic_concentration_smoothness():
    """Check that small increments produce smooth convex growth."""
    w_base = pd.Series([CONCENTRATION_THRESHOLD + 0.05])
    w_up = pd.Series([CONCENTRATION_THRESHOLD + 0.051])

    base = quadratic_concentration_addon(w_base)
    up = quadratic_concentration_addon(w_up)

    assert up > base
    assert (up - base) < 10 * base  # sanity bound: no explosion
