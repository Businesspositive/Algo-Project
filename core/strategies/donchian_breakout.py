from __future__ import annotations

import pandas as pd

from .base import Strategy


class DonchianBreakout(Strategy):
    allowed_regimes = ("TREND", "VOL_BREAKOUT", "NORMAL")

    def __init__(self, lookback_high: int = 20, lookback_low: int = 10):
        super().__init__("Donchian_Breakout")
        self.lb_h = lookback_high
        self.lb_l = lookback_low

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        required = {"High", "Low", "Close"}
        if not required.issubset(out.columns):
            raise ValueError(f"DataFrame must contain columns: {required}")
        out["DC_H"] = out["High"].rolling(self.lb_h).max().shift(1)
        out["DC_L"] = out["Low"].rolling(self.lb_l).min().shift(1)
        out["signal"] = 0
        out.loc[out["Close"] > out["DC_H"], "signal"] = 1
        out.loc[out["Close"] < out["DC_L"], "signal"] = 0
        out["signal"] = out["signal"].fillna(0).astype(int)
        out["score"] = out["signal"]
        return out


