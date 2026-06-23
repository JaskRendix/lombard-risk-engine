# Lombard Lending Risk Engine

A modular Python package for computing collateral haircuts, LTV limits, liquidity, concentration, wrong‑way‑risk adjustments, and stress‑scenario impacts for Lombard lending portfolios. The package provides a reproducible framework for evaluating credit exposure on portfolios pledged as collateral.

---

## Features

- Portfolio returns aggregation  
- EWMA volatility  
- Parametric VaR (historical)  
- Cornish–Fisher VaR (skew/kurtosis‑adjusted)  
- Expected Shortfall (historical, Gaussian, EWMA)  
- Volatility scaling utilities  
- Liquidity discounts based on ADV and position size  
- Single‑name concentration add‑on (linear or quadratic)  
- Wrong‑way‑risk multiplier (borrower–collateral correlation)  
- Combined haircut model  
- LTV and margin‑call evaluation  
- Stress scenario utilities  
- Centralized configuration parameters  

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/yourusername/lombard-risk-engine.git
cd lombard-risk-engine
pip install -e .
```

Install development dependencies:

```bash
pip install -e .[test]
```

---

## Package Structure

```
lombard_risk/
    config.py
    data.py
    collateral.py
    market_models.py
    liquidity.py
    concentration.py
    haircut.py
    ltv.py
    stress.py
    wwr.py
    engine.py
scripts/
    run_lombard_eval.py
tests/
    test_engine.py
```

---

## Basic Usage

### 1. Prepare returns and weights

```python
import pandas as pd
import numpy as np

dates = pd.date_range("2020-01-01", periods=252)
returns_df = pd.DataFrame(
    np.random.normal(0, 0.01, size=(252, 3)),
    index=dates,
    columns=["A", "B", "C"],
)

weights = pd.Series([0.5, 0.3, 0.2], index=returns_df.columns)
```

### 2. Define liquidity profiles

```python
from lombard_risk.liquidity import LiquidityProfile

liq_profiles = {
    "A": LiquidityProfile(isin="A", adv=1_000_000, position_value=300_000),
    "B": LiquidityProfile(isin="B", adv=500_000, position_value=400_000),
    "C": LiquidityProfile(isin="C", adv=200_000, position_value=300_000),
}
```

### 3. Run the engine

```python
from lombard_risk.engine import LombardRiskEngine

engine = LombardRiskEngine()

result = engine.evaluate(
    portfolio_value=1_000_000,
    loan_amount=600_000,
    returns_df=returns_df,
    weights=weights,
    base_haircut=0.20,
    liq_profiles=liq_profiles,
)
```

Optional: enable Cornish–Fisher VaR

```python
result = engine.evaluate(
    ...,
    use_cornish_fisher=True,
)
```

Optional: use quadratic concentration curve

```python
result = engine.evaluate(
    ...,
    use_quadratic_concentration=True,
)
```

Optional: apply wrong‑way‑risk multiplier

```python
result = engine.evaluate(
    ...,
    wwr_correlation=0.6,  # borrower–collateral correlation
)
```

---

## Configuration

All tunable parameters are defined in `lombard_risk/config.toml`:

- VaR confidence level  
- EWMA lambda  
- Volatility buffer  
- Liquidity thresholds and discounts  
- Concentration threshold and penalty  
- Optional quadratic concentration toggle  
- Wrong‑way‑risk lambda  
- Margin‑call buffer  
- Default base haircut  

These values can be modified to match internal policy.

---

## Testing

Run the test suite:

```bash
pytest -q
```

---

## License

MIT License.
