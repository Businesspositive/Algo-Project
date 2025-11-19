import pandas as pd
import numpy as np

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI using exponential moving average, same logic for both backtesting and daily selector.
    """
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)
