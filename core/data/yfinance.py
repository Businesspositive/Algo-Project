from __future__ import annotations

from typing import Optional

import pandas as pd
import yfinance as yf

from .base import DataSource


class YFinanceDataSource(DataSource):
    """
    Lightweight fallback data source using yfinance.
    """

    def __init__(self, auto_adjust: bool = False):
        self.auto_adjust = auto_adjust

    def fetch(
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
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        df.index.name = "Date"
        return df


