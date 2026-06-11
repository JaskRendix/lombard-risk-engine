from dataclasses import dataclass

from .config import DEFAULT_BASE_HAIRCUT


@dataclass
class CollateralRule:
    asset_class: str
    base_haircut: float


def get_base_haircut(asset_class: str, rules: list[CollateralRule]) -> float:
    for r in rules:
        if r.asset_class == asset_class:
            return r.base_haircut
    return DEFAULT_BASE_HAIRCUT
