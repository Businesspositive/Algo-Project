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

@pytest.fixture(autouse=True)
def isolated_market_data_cache(tmp_path, monkeypatch):
    """
    Point the on-disk market-data cache (core/data/base.py) at a per-test
    directory. Providers that opt in with ``cache = DEFAULT_CACHE`` resolve the
    location at call time, so this keeps tests from reading or writing the
    developer's real cache -- or each other's -- while still exercising the
    real cache code path.
    """
    cache_dir = tmp_path / "market_data_cache"
    monkeypatch.setenv("MARKET_DATA_CACHE_DIR", str(cache_dir))
    return cache_dir