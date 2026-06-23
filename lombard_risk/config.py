from pathlib import Path

import tomllib

_CONFIG_PATH = Path(__file__).with_suffix(".toml")

with _CONFIG_PATH.open("rb") as f:
    _cfg = tomllib.load(f)


def _get(key, default):
    return _cfg.get(key, default)


ALPHA = _get("ALPHA", 0.99)
EWMA_LAMBDA = _get("EWMA_LAMBDA", 0.94)
VOL_BUFFER = _get("VOL_BUFFER", 0.02)

LIQUIDITY_THRESHOLDS = _cfg.get("liquidity", {}).get("thresholds", {})
LIQUIDITY_DISCOUNTS = _cfg.get("liquidity", {}).get("discounts", {})

CONCENTRATION_THRESHOLD = _get("CONCENTRATION_THRESHOLD", 0.20)
CONCENTRATION_PENALTY = _get("CONCENTRATION_PENALTY", 0.05)

MARGIN_CALL_BUFFER = _get("MARGIN_CALL_BUFFER", 0.10)

DEFAULT_BASE_HAIRCUT = _get("DEFAULT_BASE_HAIRCUT", 0.30)
