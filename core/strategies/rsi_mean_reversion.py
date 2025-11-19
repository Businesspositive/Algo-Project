from __future__ import annotations

import pandas as pd

from .base import Strategy
from utils.rsi_utils import compute_rsi


class RSIMeanReversion(Strategy):
    allowed_regimes = ("RANGE",)

    def __init__(self, period: int = 14, lower: int = 30, upper: int = 55):
        super().__init__("RSI_MeanReversion")
        self.period = period
        self.lower = lower
        self.upper = upper

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["RSI"] = compute_rsi(out["Close"], self.period)
        out["signal"] = 0
        out.loc[out["RSI"] < self.lower, "signal"] = 1
        out.loc[out["RSI"] > self.upper, "signal"] = 0
        out["signal"] = out["signal"].astype(int)
        out["score"] = out["RSI"]
        return out


