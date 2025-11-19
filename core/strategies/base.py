from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Iterable

import pandas as pd


class Strategy(ABC):
    """
    Common contract for strategies.
    """

    name: str
    allowed_regimes: Iterable[str] = ()

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Must return a DataFrame with at least:
          - index: datetime
          - 'signal': int (1 long, 0 flat)
          - optionally 'score': diagnostic metric
        """
        raise NotImplementedError

    def latest_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        signals = self.generate_signals(df)
        latest = signals.iloc[-1]
        score = latest["score"] if "score" in signals.columns else latest.get("signal", 0)
        return {
            "signal": int(latest.get("signal", 0)),
            "score": float(score) if pd.notna(score) else 0.0,
            "raw": latest,
        }


