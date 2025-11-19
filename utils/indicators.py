# utils/indicators.py
from __future__ import annotations
import pandas as pd
import numpy as np

__all__ = [
    "true_range",
    "atr",
    "adx",
    "bollinger_bandwidth",
    "rolling_vol",
]

# --- helpers -----------------------------------------------------------------

def _require_cols(df: pd.DataFrame, cols=("High", "Low", "Close")) -> None:
    """
    Ensure df has the OHLC columns as plain strings.
    Raises ValueError if any are missing.
    """
    # Flatten any MultiIndex columns that sometimes appear from data sources
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [" ".join([c for c in tup if str(c) != ""]) for tup in df.columns]  # e.g., ('Close','') -> 'Close'

    # Coerce to strings just in case
    have = set(map(str, df.columns))
    need = set(cols)
    missing = need - have
    if missing:
        raise ValueError(f"DataFrame must contain columns: {need}. Missing: {missing}")

# --- indicators ---------------------------------------------------------------

def true_range(df: pd.DataFrame) -> pd.Series:
    """
    Wilder's True Range.
    TR_t = max(High-Low, |High - Close_{t-1}|, |Low - Close_{t-1}|).
    """
    _require_cols(df, ("High", "Low", "Close"))
    high, low, close = df["High"], df["Low"], df["Close"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return pd.to_numeric(tr, errors="coerce").rename("TR")


def atr(df: pd.DataFrame, period: int = 14, wilder: bool = True) -> pd.Series:
    """
    Average True Range.
    If wilder=True, uses Wilder's smoothing (EMA alpha=1/period). Else simple rolling mean.
    Returns a Series named 'ATR'.
    """
    print(df.columns)
    tr = true_range(df)
    if wilder:
        out = tr.ewm(alpha=1/period, adjust=False).mean()
    else:
        out = tr.rolling(window=period, min_periods=1).mean()
    return pd.to_numeric(out, errors="coerce").fillna(0.0).rename("ATR")


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index (simplified, robust).
    Returns a Series in 0..100 named 'ADX'.
    """
    _require_cols(df, ("High", "Low", "Close"))
    high, low = df["High"], df["Low"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(df)
    # Wilder-style smoothing of DM and TR
    sm_tr = tr.ewm(alpha=1/period, adjust=False).mean()
    sm_plus_dm = pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    sm_minus_dm = pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()

    plus_di = 100.0 * (sm_plus_dm / sm_tr.replace(0, np.nan))
    minus_di = 100.0 * (sm_minus_dm / sm_tr.replace(0, np.nan))

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100.0
    adx_series = dx.ewm(alpha=1/period, adjust=False).mean()

    return pd.to_numeric(adx_series, errors="coerce").fillna(0.0).clip(lower=0.0, upper=100.0).rename("ADX")


def bollinger_bandwidth(series: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """
    Bollinger Bandwidth = (Upper - Lower) / Middle.
    Middle = rolling mean. Upper/Lower = Middle ± num_std * rolling std.
    Returns Series named 'BBWidth'.
    """
    ma = series.rolling(window=period, min_periods=1).mean()
    std = series.rolling(window=period, min_periods=1).std(ddof=0)
    upper = ma + num_std * std
    lower = ma - num_std * std
    width = (upper - lower) / ma.replace(0, np.nan)
    return pd.to_numeric(width, errors="coerce").fillna(0.0).rename("BBWidth")


def rolling_vol(series: pd.Series, period: int = 20, log: bool = True) -> pd.Series:
    """
    Rolling volatility (std of returns) over `period`.
    If log=True, uses log returns; else percentage returns.
    Returns Series named 'RVol'.
    """
    if log:
        rets = np.log(series / series.shift(1))
    else:
        rets = series.pct_change()
    vol = rets.rolling(window=period, min_periods=2).std(ddof=0)
    return pd.to_numeric(vol, errors="coerce").fillna(0.0).rename("RVol")
