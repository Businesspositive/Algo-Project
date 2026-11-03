import numpy as np
import pandas as pd
import pytest

from core.data.base import DataSource, OHLCV_COLUMNS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class ReturnsWhatever(DataSource):
    """Provider double whose raw payload is injected by the test."""

    def __init__(self, payload):
        self.payload = payload

    def _fetch_raw(self, symbol, start, end=None, interval="1d"):
        return self.payload


def ohlcv(index) -> pd.DataFrame:
    n = len(index)
    return pd.DataFrame(
        {
            "Open": np.arange(n, dtype=float),
            "High": np.arange(n, dtype=float) + 1,
            "Low": np.arange(n, dtype=float) - 1,
            "Close": np.arange(n, dtype=float),
            "Volume": 100,
        },
        index=index,
    )


def fetch(payload) -> pd.DataFrame:
    return ReturnsWhatever(payload).fetch("XYZ", "2025-01-01")


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------
def test_naive_index_is_localised_to_utc():
    naive = pd.date_range("2025-01-01", periods=5, freq="D")
    out = fetch(ohlcv(naive))

    assert isinstance(out.index, pd.DatetimeIndex)
    assert str(out.index.tz) == "UTC"
    # Naive timestamps are interpreted *as* UTC: wall-clock values unchanged.
    assert list(out.index.tz_localize(None)) == list(naive)


def test_aware_non_utc_index_is_converted_to_utc():
    ist = pd.date_range("2025-01-01 09:15", periods=3, freq="h", tz="Asia/Kolkata")
    out = fetch(ohlcv(ist))

    assert str(out.index.tz) == "UTC"
    # Same instants, expressed in UTC (IST = UTC+05:30).
    assert out.index[0] == pd.Timestamp("2025-01-01 03:45", tz="UTC")
    assert (out.index == ist).all()


def test_already_utc_index_is_unchanged():
    utc = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    out = fetch(ohlcv(utc))
    assert out.index.equals(utc)


def test_columns_are_returned_in_canonical_order():
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    shuffled = ohlcv(idx)[["Volume", "Close", "Low", "High", "Open"]]
    out = fetch(shuffled)
    assert list(out.columns) == list(OHLCV_COLUMNS)


def test_empty_frame_helper_satisfies_contract():
    out = fetch(DataSource.empty_frame())
    assert out.empty
    assert list(out.columns) == list(OHLCV_COLUMNS)
    assert isinstance(out.index, pd.DatetimeIndex)
    assert str(out.index.tz) == "UTC"


def test_provider_frame_is_not_mutated():
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    raw = ohlcv(idx)
    fetch(raw)
    assert raw.index.tz is None  # caller's object untouched


# ---------------------------------------------------------------------------
# Column schema violations
# ---------------------------------------------------------------------------
def test_missing_column_raises_value_error():
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    raw = ohlcv(idx).drop(columns=["Volume"])
    with pytest.raises(ValueError, match=r"missing required columns \['Volume'\]"):
        fetch(raw)


def test_extra_column_raises_value_error():
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    raw = ohlcv(idx)
    raw["Adj Close"] = raw["Close"]
    with pytest.raises(ValueError, match=r"unexpected columns \['Adj Close'\]"):
        fetch(raw)


def test_multiindex_columns_raise_value_error():
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    raw = ohlcv(idx)
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["XYZ"]])
    with pytest.raises(ValueError, match="MultiIndex"):
        fetch(raw)


def test_duplicate_columns_raise_value_error():
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    raw = ohlcv(idx)
    raw = pd.concat([raw, raw[["Close"]]], axis=1)
    with pytest.raises(ValueError, match=r"duplicate columns \['Close'\]"):
        fetch(raw)


def test_bare_empty_dataframe_is_rejected():
    # The old providers returned pd.DataFrame() when there was no data; that
    # has no columns and must now go through DataSource.empty_frame().
    with pytest.raises(ValueError, match="missing required columns"):
        fetch(pd.DataFrame())


def test_non_dataframe_return_is_rejected():
    with pytest.raises(ValueError, match="must return a pandas DataFrame"):
        fetch(None)


# ---------------------------------------------------------------------------
# Index violations
# ---------------------------------------------------------------------------
def test_non_datetime_index_raises_value_error():
    raw = ohlcv(pd.RangeIndex(3))
    with pytest.raises(ValueError, match="must be a pandas DatetimeIndex"):
        fetch(raw)


def test_string_index_raises_value_error():
    raw = ohlcv(pd.Index(["2025-01-01", "2025-01-02", "2025-01-03"]))
    with pytest.raises(ValueError, match="must be a pandas DatetimeIndex"):
        fetch(raw)


def test_nat_in_index_raises_value_error():
    idx = pd.DatetimeIndex([pd.Timestamp("2025-01-01"), pd.NaT, pd.Timestamp("2025-01-03")])
    with pytest.raises(ValueError, match="NaT"):
        fetch(ohlcv(idx))


# ---------------------------------------------------------------------------
# Error messages identify the provider and symbol
# ---------------------------------------------------------------------------
def test_error_message_names_provider_and_symbol():
    with pytest.raises(ValueError, match=r"ReturnsWhatever\.fetch\('XYZ'\)"):
        fetch(pd.DataFrame())


# ---------------------------------------------------------------------------
# Contract cannot be bypassed by overriding fetch()
# ---------------------------------------------------------------------------
def test_overriding_fetch_is_rejected_at_class_creation():
    with pytest.raises(TypeError, match="must not override DataSource.fetch"):

        class Sneaky(DataSource):
            def fetch(self, symbol, start, end=None, interval="1d"):
                return pd.DataFrame()

            def _fetch_raw(self, symbol, start, end=None, interval="1d"):
                return pd.DataFrame()


# ---------------------------------------------------------------------------
# Concrete providers (network mocked out)
# ---------------------------------------------------------------------------
def test_yfinance_provider_standardises_output(monkeypatch):
    yfmod = pytest.importorskip("core.data.yfinance")

    def fake_download(*args, **kwargs):
        idx = pd.date_range("2025-01-01", periods=3, freq="D", name="Date")
        df = ohlcv(idx)
        df["Adj Close"] = df["Close"]  # yfinance extra column must be dropped
        df.columns = pd.MultiIndex.from_product([df.columns, ["XYZ"]])  # yfinance>=0.2 shape
        return df

    monkeypatch.setattr(yfmod.yf, "download", fake_download)
    out = yfmod.YFinanceDataSource().fetch("XYZ", "2025-01-01")

    assert list(out.columns) == list(OHLCV_COLUMNS)
    assert str(out.index.tz) == "UTC"
    assert len(out) == 3


def test_yfinance_provider_empty_download_returns_schema_frame(monkeypatch):
    yfmod = pytest.importorskip("core.data.yfinance")
    monkeypatch.setattr(yfmod.yf, "download", lambda *a, **k: pd.DataFrame())
    out = yfmod.YFinanceDataSource().fetch("XYZ", "2025-01-01")
    assert out.empty and list(out.columns) == list(OHLCV_COLUMNS)
    assert str(out.index.tz) == "UTC"


def test_yfinance_provider_missing_column_is_clear_error(monkeypatch):
    yfmod = pytest.importorskip("core.data.yfinance")
    idx = pd.date_range("2025-01-01", periods=3, freq="D")
    monkeypatch.setattr(yfmod.yf, "download", lambda *a, **k: ohlcv(idx).drop(columns=["High"]))
    with pytest.raises(ValueError, match=r"YFinanceDataSource\.fetch\('XYZ'\).*missing required columns \['High'\]"):
        yfmod.YFinanceDataSource().fetch("XYZ", "2025-01-01")


def test_fyers_provider_standardises_output():
    fymod = pytest.importorskip("core.data.fyers")

    class FakeClient:
        def history(self, payload):
            # epoch seconds -> naive UTC after pd.to_datetime(unit="s")
            base = int(pd.Timestamp("2025-01-01", tz="UTC").timestamp())
            return {
                "candles": [
                    [base + i * 86400, 10.0, 11.0, 9.0, 10.5, 1000] for i in range(3)
                ]
            }

    out = fymod.FyersDataSource(fyers_client=FakeClient()).fetch("NSE:SBIN-EQ", "2025-01-01", "2025-01-03")

    assert list(out.columns) == list(OHLCV_COLUMNS)
    assert str(out.index.tz) == "UTC"
    assert out.index[0] == pd.Timestamp("2025-01-01", tz="UTC")


def test_fyers_provider_no_candles_returns_schema_frame():
    fymod = pytest.importorskip("core.data.fyers")

    class EmptyClient:
        def history(self, payload):
            return {"candles": []}

    out = fymod.FyersDataSource(fyers_client=EmptyClient()).fetch("NSE:SBIN-EQ", "2025-01-01", "2025-01-03")
    assert out.empty and list(out.columns) == list(OHLCV_COLUMNS)
    assert str(out.index.tz) == "UTC"