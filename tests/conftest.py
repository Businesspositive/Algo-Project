import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv():
    dates = pd.date_range(
        start="2025-01-01",
        periods=250,
        freq="B",
    )

    close = pd.Series(
        range(100, 350),
        index=dates,
        dtype=float,
    )

    return pd.DataFrame(
        {
            "Open": close - 1,
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Volume": 10000,
        },
        index=dates,
    )