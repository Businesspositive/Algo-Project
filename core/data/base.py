from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


class DataSource(ABC):
    """
    Minimal contract every market data provider must satisfy.
    """

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        start: str,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Return an OHLCV DataFrame indexed by datetime with columns
        ['Open','High','Low','Close','Volume'].
        """
        raise NotImplementedError


