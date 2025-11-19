from typing import Dict, List, Optional

import pandas as pd
from datetime import datetime

from core.data import DataSource, FyersDataSource
from core.strategies import strategies_for_regime
from regimes.detector import RegimeDetector, RegimeThresholds


class DailySelector:
    def __init__(
        self,
        universe: List[str],
        start: str,
        end: Optional[str] = None,
        data_source: Optional[DataSource] = None,
        thresholds: RegimeThresholds = RegimeThresholds(),
    ):
        self.universe = universe
        self.start = start
        self.end = end or datetime.today().strftime("%Y-%m-%d")
        self.data_source = data_source or FyersDataSource()
        self.detector = RegimeDetector(thresholds)

    def _fetch(self, ticker: str) -> pd.DataFrame:
        try:
            df = self.data_source.fetch(ticker, self.start, self.end, interval="D")
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            return df
        except Exception as exc:
            print(f"[FETCH ERROR] {ticker}: {exc}")
            return pd.DataFrame()

    def pick(self) -> Dict[str, Dict[str, object]]:
        picks: Dict[str, Dict[str, object]] = {}
        for ticker in self.universe:
            df = self._fetch(ticker)
            if df.empty or len(df) < 50:
                continue

            try:
                regime_info = self.detector.latest_regime(df)
            except Exception as exc:
                print(f"[REGIME ERROR] {ticker}: {exc}")
                continue
            regime = regime_info["regime"]

            strategies = strategies_for_regime(regime)
            if not strategies:
                continue

            results = []
            for strategy in strategies:
                try:
                    signal_info = strategy.latest_signal(df)
                except Exception as exc:
                    print(f"[STRATEGY ERROR] {ticker} {strategy.name}: {exc}")
                    continue
                results.append(
                    {
                        "name": strategy.name,
                        "signal": signal_info["signal"],
                        "score": signal_info["score"],
                    }
                )

            if not results:
                continue

            picks[ticker] = {
                "regime": regime,
                "strategies": results,
            }
        return picks

