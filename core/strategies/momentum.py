from __future__ import annotations

import pandas as pd

from .base import Strategy

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


class MomentumStrategy(Strategy):
    allowed_regimes = ("TREND", "NORMAL")

    def __init__(self, roc_period: int = 12, roc_thresh: float = 0.02, ema_period: int = 50):
        super().__init__("Momentum")
        self.roc_period = roc_period
        self.roc_thresh = roc_thresh
        self.ema_period = ema_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["ROC"] = out["Close"].pct_change(self.roc_period)
        ema_series = _ema(out["Close"], self.ema_period)
        ema_series = ema_series.reindex(out.index).bfill()
        out["EMA"] = ema_series
        out["signal"] = 0
        cond = (out["ROC"] > self.roc_thresh) & (out["Close"] > out["EMA"])
        out.loc[cond, "signal"] = 1
        out["signal"] = out["signal"].astype(int)
        out["score"] = out["ROC"]
        return out

