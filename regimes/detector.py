# regimes/detector.py
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any

from utils.indicators import atr, adx, bollinger_bandwidth, rolling_vol


@dataclass
class RegimeThresholds:
    """Configurable thresholds for regime classification.
    Tune per market (India vs US) or timeframe (daily vs hourly).
    """
    adx_trend: float = 20.0       # ADX above ⇒ trending
    bb_width_low: float = 0.06    # BBWidth below ⇒ compressed / range
    vol_high: float = 0.025       # Rolling vol above ⇒ high-volatility
    atr_floor_frac: float = 0.004 # ATR/Close below ⇒ too quiet (filter out)


class RegimeDetector:
    """Computes market regime labels for a price series.

    Regimes:
    - "TREND": persistent directional moves (ADX high)
    - "RANGE": compressed volatility / mean-reverting behavior (BBWidth low)
    - "VOL_BREAKOUT": high vol expansion (Rolling vol high)
    - "QUIET": too low ATR/Close (often avoid trading)

    Notes:
    * Priority order matters. We label QUIET first (avoid), then VOL, then TREND, then RANGE.
    * You can extend with event filters (earnings, macro days) upstream.
    """

    def __init__(
        self,
        thresholds: RegimeThresholds | None = None,
        bb_period: int = 20,
        vol_period: int = 20,
        atr_period: int = 14,
        adx_period: int = 14,
    ):
        self.thr = thresholds or RegimeThresholds()
        self.bb_period = bb_period
        self.vol_period = vol_period
        self.atr_period = atr_period
        self.adx_period = adx_period

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # --- ATR ---
        try:
            atr_vals = atr(out, period=self.atr_period)
            if isinstance(atr_vals, pd.DataFrame):
                atr_vals = atr_vals.iloc[:, 0]
            out["ATR"] = atr_vals
        except Exception as e:
            print(f"Error calculating ATR: {e}")
            out["ATR"] = pd.Series([pd.NA] * len(out), index=out.index)

        # --- ADX ---
        try:
            adx_vals = adx(out, period=self.adx_period)
            if isinstance(adx_vals, pd.DataFrame):
                adx_vals = adx_vals.iloc[:, 0]
            out["ADX"] = adx_vals
        except Exception as e:
            print(f"Error calculating ADX: {e}")
            out["ADX"] = pd.Series([pd.NA] * len(out), index=out.index)

        # --- Bollinger Bandwidth ---
        try:
            out["BBWidth"] = bollinger_bandwidth(out["Close"], period=self.bb_period)
        except Exception as e:
            print(f"Error calculating BBWidth: {e}")
            out["BBWidth"] = pd.Series([pd.NA] * len(out), index=out.index)

        # --- Rolling Volatility ---
        try:
            out["RVol"] = rolling_vol(out["Close"], period=self.vol_period)
        except Exception as e:
            print(f"Error calculating Rolling Vol: {e}")
            out["RVol"] = pd.Series([pd.NA] * len(out), index=out.index)

        # --- ATR / Close ---
        out["ATR_frac"] = out["ATR"] / out["Close"].replace(0, pd.NA)

        return out


    def classify_row(self, row: pd.Series) -> str:
        # Order: QUIET → VOL_BREAKOUT → TREND → RANGE → NORMAL
        if row["ATR_frac"] < self.thr.atr_floor_frac:
            return "QUIET"
        if row["RVol"] >= self.thr.vol_high:
            return "VOL_BREAKOUT"
        if row["ADX"] >= self.thr.adx_trend:
            return "TREND"
        if row["BBWidth"] <= self.thr.bb_width_low:
            return "RANGE"
        return "NORMAL"

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        feats = self.compute_features(df)
        feats["Regime"] = feats.apply(self.classify_row, axis=1)
        return feats

    def latest_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        feats = self.detect(df)
        last = feats.iloc[-1]
        return {
            "regime": last["Regime"],
            "adx": float(last["ADX"]),
            "bb_width": float(last["BBWidth"]),
            "rvol": float(last["RVol"]),
            "atr_frac": float(last["ATR_frac"]),
        }
