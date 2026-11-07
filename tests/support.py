"""
Shared, deterministic test doubles for the routing / forced-flatten /
settlement tests.  No network, no clock dependence beyond pd.Timestamp.now().
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import orchestrator
from brokers import FyersPaperBroker
from core.data.base import DataSource
from regimes.detector import RegimeDetector


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
def trend_frame(last_bar: str = "breakout", periods: int = 120) -> pd.DataFrame:
    """
    A steadily rising series whose *last* bar decides the strategies' signals:

      "breakout" -> Close jumps above the prior 20-day high:
                    Donchian_Breakout = 1, Momentum = 1, RSI_MeanReversion = 0
      "collapse" -> Close drops below the prior 10-day low:
                    Donchian_Breakout = 0
      "flat"     -> continues the ramp; never breaks the High=Close+2 band:
                    Donchian_Breakout = 0
    """
    idx = pd.date_range("2025-01-01", periods=periods, freq="B")
    close = pd.Series(np.linspace(100.0, 200.0, periods), index=idx)
    df = pd.DataFrame(
        {"Open": close - 1, "High": close + 2, "Low": close - 2, "Close": close, "Volume": 1000},
        index=idx,
    )
    if last_bar == "breakout":
        df.iloc[-1] = [230.0, 232.0, 228.0, 230.0, 1000]
    elif last_bar == "collapse":
        df.iloc[-1] = [150.0, 151.0, 149.0, 150.0, 1000]
    elif last_bar != "flat":
        raise ValueError(last_bar)
    return df


class StaticDataSource(DataSource):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def _fetch_raw(self, symbol, start, end=None, interval="1d"):
        return self.df


# ---------------------------------------------------------------------------
# Regime detectors
# ---------------------------------------------------------------------------
class FixedRegimeDetector(RegimeDetector):
    """
    Scripted detector.

      FixedRegimeDetector("TREND")
          every bar in the window is TREND (a long-stable regime).
      FixedRegimeDetector(history=["TREND", "TREND", "RANGE"])
          the *tail* of the per-bar history is as given (latest last); earlier
          bars repeat the first element.  `latest_regime` reports the last.
      FixedRegimeDetector(None)
          raises from both methods, like a broken detector.
    """

    def __init__(self, regime: Optional[str] = "TREND", history: Optional[list] = None):
        super().__init__()
        if history is not None:
            self.history = list(history)
            self.regime = self.history[-1]
        else:
            self.history = None if regime is None else [regime]
            self.regime = regime

    def latest_regime(self, df):
        if self.regime is None:
            raise RuntimeError("detector exploded")
        return {"regime": self.regime, "adx": 0.0, "bb_width": 0.0, "rvol": 0.0, "atr_frac": 0.0}

    def regime_history(self, df):
        if self.history is None:
            raise RuntimeError("detector exploded")
        n = len(df)
        tail = self.history[-n:]
        labels = [self.history[0]] * (n - len(tail)) + tail
        return pd.Series(labels, index=df.index, name="Regime")


# ---------------------------------------------------------------------------
# Broker
# ---------------------------------------------------------------------------
class ScriptedBroker(FyersPaperBroker):
    """
    Paper broker with a mutable price table plus knobs to script how SELL
    orders behave, so tests can simulate an exchange that rejects or only
    partially fills a forced flatten:

      reject_sells      -> every SELL returns {"status": "rejected"} and
                           leaves the position untouched.
      partial_sell_qty  -> a SELL fills only this many shares and reports an
                           exchange-style "PARTIALLY_FILLED" status.
    """

    def __init__(self, prices: Dict[str, float] | float = 230.0, cash: float = 100_000.0,
                 fees_pct: float = 0.0):
        self.prices: Dict[str, float] = {}
        self.default_price = float(prices) if not isinstance(prices, dict) else None
        if isinstance(prices, dict):
            self.prices.update(prices)
        super().__init__(initial_cash=cash, market_price_provider=self._price, fees_pct=fees_pct)
        self.reject_sells = False
        self.partial_sell_qty: Optional[float] = None
        self.orders: list[tuple[str, str, float]] = []  # every attempt, in order

    def _price(self, symbol: str) -> float:
        if symbol in self.prices:
            return self.prices[symbol]
        if self.default_price is None:
            raise KeyError(symbol)
        return self.default_price

    def place_order(self, symbol: str, side: str, qty: float, **kwargs: Any) -> Dict[str, Any]:
        self.orders.append((symbol, str(side).upper(), float(qty)))
        if str(side).upper() == "SELL":
            if self.reject_sells:
                return {"status": "rejected", "reason": "exchange-rejected"}
            if self.partial_sell_qty is not None and self.partial_sell_qty < qty:
                res = super().place_order(symbol, side, self.partial_sell_qty, **kwargs)
                res["status"] = "PARTIALLY_FILLED"
                return res
        return super().place_order(symbol, side, qty, **kwargs)

    def sells(self):
        return [o for o in self.orders if o[1] == "SELL"]

    def buys(self):
        return [o for o in self.orders if o[1] == "BUY"]


# ---------------------------------------------------------------------------
# Trade-log helpers
# ---------------------------------------------------------------------------
def seed_healthy_history(strategy_name: str, n: int = 12) -> None:
    """Enough profitable closed trades that `strategy_name` passes the health gate."""
    now = pd.Timestamp.now()
    for i in range(n):
        orchestrator.append_trade_to_history(
            {
                "timestamp": now - pd.Timedelta(days=n - i),
                "strategy": strategy_name,
                "symbol": "SEED",
                "side": "SELL",
                "qty": 1,
                "price": 100.0,
                "entry_price": 95.0,
                "fees": 0.0,
                "pnl": 5.0 + i,
                "reason": orchestrator.REASON_SIGNAL,
            }
        )


def symbol_rows(symbol: str) -> pd.DataFrame:
    """Trade-log rows for one symbol, in log order (excludes SEED rows)."""
    df = orchestrator.load_trade_history()
    return df[df["symbol"] == symbol].reset_index(drop=True)