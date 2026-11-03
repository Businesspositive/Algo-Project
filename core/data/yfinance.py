from __future__ import annotations

from typing import Optional

import pandas as pd
import yfinance as yf

from .base import DataSource, DEFAULT_CACHE, OHLCV_COLUMNS


class YFinanceDataSource(DataSource):
    """
    Lightweight fallback data source using yfinance.
    """

    #: Remote API: cache standardised bars locally to stay clear of rate limits.
    cache = DEFAULT_CACHE

    def __init__(self, auto_adjust: bool = False):
        self.auto_adjust = auto_adjust

    def _fetch_raw(
        self,
        symbol: str,
        start: str,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            progress=False,
            auto_adjust=self.auto_adjust,
        )
        if df is None or df.empty:
            return self.empty_frame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Keep only the OHLCV columns that exist (yfinance adds e.g. 'Adj Close');
        # any that are missing are reported by DataSource.standardise().
        keep = [c for c in OHLCV_COLUMNS if c in df.columns]
        df = df[keep].dropna()
        df.index.name = "Date"
        return df