"""
Central configuration for Lombard Lending Risk Engine.
All tunable risk parameters live here.
"""

# --- Market Risk ---
ALPHA = 0.99  # VaR/ES confidence level
EWMA_LAMBDA = 0.94  # EWMA decay factor
VOL_BUFFER = 0.02  # extra haircut buffer on top of VaR

# --- Liquidity Risk ---
LIQUIDITY_THRESHOLDS = {
    "low": 0.1,  # position_value / ADV < 0.1 → no discount
    "medium": 0.5,  # < 0.5 → 2%
    "high": 1.0,  # < 1.0 → 5%
}
LIQUIDITY_DISCOUNTS = {
    "low": 0.00,
    "medium": 0.02,
    "high": 0.05,
    "extreme": 0.10,  # > 1.0 ADV → 10%
}

# --- Concentration Risk ---
CONCENTRATION_THRESHOLD = 0.20  # 20% max single-name weight
CONCENTRATION_PENALTY = 0.05  # 5% haircut per 10% above threshold

# --- Margin Call ---
MARGIN_CALL_BUFFER = 0.10  # 10% above max LTV triggers margin call

# --- Collateral Eligibility ---
DEFAULT_BASE_HAIRCUT = 0.30  # fallback haircut if asset class unknown
