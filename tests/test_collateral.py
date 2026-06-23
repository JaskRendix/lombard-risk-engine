import pytest

from lombard_risk.collateral import CollateralRule, get_base_haircut
from lombard_risk.config import DEFAULT_BASE_HAIRCUT


def test_get_base_haircut_exact_match():
    rules = [
        CollateralRule("EQUITY", 0.20),
        CollateralRule("BOND", 0.10),
    ]
    assert get_base_haircut("EQUITY", rules) == 0.20
    assert get_base_haircut("BOND", rules) == 0.10


def test_get_base_haircut_fallback_default():
    rules = [CollateralRule("EQUITY", 0.20)]
    assert get_base_haircut("REAL_ESTATE", rules) == DEFAULT_BASE_HAIRCUT


def test_get_base_haircut_empty_rules():
    assert get_base_haircut("ANYTHING", []) == DEFAULT_BASE_HAIRCUT


def test_get_base_haircut_first_match_wins():
    rules = [
        CollateralRule("EQUITY", 0.20),
        CollateralRule("EQUITY", 0.50),  # should be ignored
    ]
    assert get_base_haircut("EQUITY", rules) == 0.20


@pytest.mark.parametrize(
    "asset_class,rules,expected",
    [
        ("FX", [CollateralRule("FX", 0.15)], 0.15),
        ("CASH", [CollateralRule("CASH", 0.01)], 0.01),
        ("UNKNOWN", [CollateralRule("BOND", 0.10)], DEFAULT_BASE_HAIRCUT),
    ],
)
def test_get_base_haircut_parametrized(asset_class, rules, expected):
    assert get_base_haircut(asset_class, rules) == expected


def test_get_base_haircut_case_sensitive():
    rules = [CollateralRule("Equity", 0.25)]
    # Should NOT match "EQUITY"
    assert get_base_haircut("EQUITY", rules) == DEFAULT_BASE_HAIRCUT
