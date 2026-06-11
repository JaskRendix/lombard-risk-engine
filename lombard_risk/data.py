from dataclasses import dataclass

import pandas as pd


@dataclass
class Position:
    isin: str
    asset_class: str
    currency: str
    quantity: float


@dataclass
class Portfolio:
    positions: list[Position]


@dataclass
class MarketSeries:
    prices: pd.Series  # index: date
    returns: pd.Series  # index: date
