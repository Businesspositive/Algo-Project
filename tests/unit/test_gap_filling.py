"""
Tests for the gap-filling ("imputation") stage of core/data/base.py.

Three properties are proven here:

1. A gap in time inserts a forward-filled *flat* bar (O = H = L = C = last
   observed close) at exactly the missing timestamps and nowhere else.
2. Every synthetic bar carries Volume == 0; observed bars are untouched.
3. Timezone handling is exact and never invents weekend candles -- in
   particular for Fyers-style daily bars stamped at 00:00 IST, which sit on
   the *previous* calendar day in UTC.
"""
import numpy as np
import pandas as pd
import pytest

from core.data.base import DataSource, OHLCV_COLUMNS

IST = "Asia/Kolkata"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class Provider(DataSource):
    """In-memory provider; ``calendar_tz`` configurable per instance."""

    def __init__(self, df: pd.DataFrame, calendar_tz: str = "UTC"):
        self.df = df
        self.calendar_tz = calendar_tz

    def _fetch_raw(self, symbol, start, end=None, interval="1d"):
        return self.df


def bars(index: pd.DatetimeIndex, *, volume: int = 1_000) -> pd.DataFrame:
    """Deterministic, strictly increasing closes so forward-fill is provable."""
    n = len(index)
    close = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(n, volume, dtype="int64"),
        },
        index=index,
    )


def ist_session(day: str) -> pd.DatetimeIndex:
    """NSE cash session, 1-minute bars, 09:15 .. 15:29 IST."""
    return pd.date_range(f"{day} 09:15", f"{day} 15:29", freq="1min", tz=IST)


def run(raw: pd.DataFrame, calendar_tz: str = "UTC") -> pd.DataFrame:
    return Provider(raw, calendar_tz).fetch("XYZ", "2025-01-01")


def synthetic_rows(out: pd.DataFrame, raw: pd.DataFrame) -> pd.DatetimeIndex:
    return out.index.difference(raw.index.tz_convert("UTC") if raw.index.tz else raw.index.tz_localize("UTC"))


# ===========================================================================
# 1. A gap in time inserts a forward-filled flat bar
# ===========================================================================
class TestGapInsertsForwardFilledFlatBar:
    def test_intraday_gap_filled_at_exact_missing_timestamps_with_last_close(self):
        full = ist_session("2025-01-06")
        missing = pd.DatetimeIndex(
            [pd.Timestamp("2025-01-06 10:00", tz=IST),   # 3-bar run
             pd.Timestamp("2025-01-06 10:01", tz=IST),
             pd.Timestamp("2025-01-06 10:02", tz=IST),
             pd.Timestamp("2025-01-06 14:30", tz=IST)]   # isolated bar
        )
        raw = bars(full.difference(missing))

        out = run(raw, calendar_tz=IST)

        # Exactly the missing minutes were inserted -- no more, no fewer.
        added = synthetic_rows(out, raw)
        assert added.equals(missing.tz_convert("UTC"))
        assert len(out) == len(full)

        # Forward fill: the whole 10:00-10:02 run carries the 09:59 close,
        # 14:30 carries the 14:29 close. Never the *next* observed close.
        close_0959 = raw.loc[pd.Timestamp("2025-01-06 09:59", tz=IST), "Close"]
        close_1003 = raw.loc[pd.Timestamp("2025-01-06 10:03", tz=IST), "Close"]
        close_1429 = raw.loc[pd.Timestamp("2025-01-06 14:29", tz=IST), "Close"]
        run_rows = out.loc[missing[:3].tz_convert("UTC")]
        assert (run_rows["Close"] == close_0959).all()
        assert (run_rows["Close"] != close_1003).all()
        assert out.loc[missing[3].tz_convert("UTC"), "Close"] == close_1429

        # Flat bar: Open == High == Low == Close on every synthetic row.
        synth = out.loc[added]
        for col in ("Open", "High", "Low"):
            assert (synth[col] == synth["Close"]).all(), col

    def test_daily_gap_filled_with_previous_days_close(self):
        days = pd.bdate_range("2025-01-06", "2025-01-17", tz="UTC")
        wednesday = pd.Timestamp("2025-01-08", tz="UTC")
        raw = bars(days.drop(wednesday))

        out = run(raw)

        assert synthetic_rows(out, raw).equals(pd.DatetimeIndex([wednesday]))
        tuesday_close = raw.loc[pd.Timestamp("2025-01-07", tz="UTC"), "Close"]
        thursday_close = raw.loc[pd.Timestamp("2025-01-09", tz="UTC"), "Close"]
        filled = out.loc[wednesday]
        assert filled["Close"] == tuesday_close != thursday_close
        assert filled["Open"] == filled["High"] == filled["Low"] == tuesday_close

    def test_observed_bars_are_returned_unmodified(self):
        full = ist_session("2025-01-06")
        raw = bars(full.delete([10, 11, 200]))

        out = run(raw, calendar_tz=IST)

        expected = DataSource.standardise(raw)
        observed = out.loc[expected.index]
        pd.testing.assert_frame_equal(observed, expected)

    def test_no_gap_means_no_change(self):
        raw = bars(ist_session("2025-01-06"))
        out = run(raw, calendar_tz=IST)
        pd.testing.assert_frame_equal(out, DataSource.standardise(raw))

    def test_output_keeps_strict_ohlcv_schema(self):
        raw = bars(ist_session("2025-01-06").delete([5]))
        out = run(raw, calendar_tz=IST)
        assert list(out.columns) == list(OHLCV_COLUMNS)
        assert not out.isna().any().any()


# ===========================================================================
# 2. Volume of every synthetic flat bar is zero
# ===========================================================================
class TestSyntheticBarVolumeIsZero:
    def test_all_synthetic_rows_have_zero_volume_observed_rows_keep_theirs(self):
        full = ist_session("2025-01-06")
        raw = bars(full.delete([30, 31, 32, 150, 300]), volume=987_654)

        out = run(raw, calendar_tz=IST)
        added = synthetic_rows(out, raw)

        assert len(added) == 5
        assert (out.loc[added, "Volume"] == 0).all()
        assert (out.drop(added)["Volume"] == 987_654).all()
        # Zero volume only where a bar was invented.
        assert set(out.index[out["Volume"] == 0]) == set(added)

    def test_zero_volume_on_daily_fill_and_integer_dtype_preserved(self):
        days = pd.bdate_range("2025-01-06", "2025-01-17", tz="UTC")
        raw = bars(days.drop(pd.Timestamp("2025-01-08", tz="UTC")))
        assert raw["Volume"].dtype == "int64"

        out = run(raw)

        assert out.loc[pd.Timestamp("2025-01-08", tz="UTC"), "Volume"] == 0
        assert out["Volume"].dtype == "int64"  # reindex() upcasts to float; we restore it

    def test_zero_volume_with_float_volume_provider(self):
        days = pd.bdate_range("2025-01-06", "2025-01-17", tz="UTC")
        raw = bars(days.drop(pd.Timestamp("2025-01-08", tz="UTC")))
        raw["Volume"] = raw["Volume"].astype("float64")

        out = run(raw)

        assert out.loc[pd.Timestamp("2025-01-08", tz="UTC"), "Volume"] == 0.0
        assert out["Volume"].dtype == "float64"


# ===========================================================================
# 3. Timezone conversion is exact and never invents weekend candles
# ===========================================================================
class TestTimezoneAndWeekends:
    def test_ist_midnight_daily_bars_fill_weekday_gap_only(self):
        """
        Fyers stamps daily candles at 00:00 IST == 18:30 UTC the previous day.
        A business-day grid evaluated in UTC would see Fri as Thu and Mon as
        Sun, and would invent Saturday bars. With calendar_tz=IST it must fill
        exactly the missing Wednesday and nothing else.
        """
        business_days_ist = pd.bdate_range("2025-01-06", "2025-01-24", tz=IST)      # 15 sessions
        missing_wed = pd.Timestamp("2025-01-15", tz=IST)
        # Emulate the provider: epoch seconds -> naive UTC (what Fyers gives us).
        epoch = business_days_ist.drop(missing_wed).tz_convert("UTC").tz_localize(None)
        raw = bars(epoch)
        assert raw.index.tz is None and raw.index[0] == pd.Timestamp("2025-01-05 18:30")

        out = run(raw, calendar_tz=IST)

        # Exactly one bar added, at the missing Wednesday's IST midnight,
        # which is Tuesday 18:30 UTC on the wire.
        added = out.index.difference(raw.index.tz_localize("UTC"))
        assert len(added) == 1
        assert added[0] == pd.Timestamp("2025-01-14 18:30", tz="UTC")
        assert added[0].tz_convert(IST) == missing_wed

        # No Saturday/Sunday (in IST) anywhere in the output, and the set of
        # IST trading dates is precisely the business-day calendar.
        ist_dates = out.index.tz_convert(IST)
        assert (ist_dates.day_of_week < 5).all()
        assert ist_dates.equals(business_days_ist)
        assert len(out) == 15

    def test_utc_midnight_daily_bars_fill_weekday_gap_only(self):
        """yfinance-style stamps (naive midnight == UTC midnight), default calendar."""
        business_days = pd.bdate_range("2025-01-06", "2025-01-24")
        missing_thu = pd.Timestamp("2025-01-16")
        raw = bars(business_days.drop(missing_thu))

        out = run(raw)  # calendar_tz="UTC"

        added = out.index.difference(raw.index.tz_localize("UTC"))
        assert added.equals(pd.DatetimeIndex([missing_thu.tz_localize("UTC")]))
        assert (out.index.day_of_week < 5).all()
        assert len(out) == len(business_days)

    def test_weekend_gap_between_friday_and_monday_is_not_filled(self):
        """A series with no missing weekday must come back exactly as given."""
        days = pd.bdate_range("2025-01-06", "2025-01-24", tz=IST).tz_convert("UTC")
        raw = bars(days)
        out = run(raw, calendar_tz=IST)
        assert out.index.equals(days)
        assert (out["Volume"] > 0).all()  # nothing synthetic

    def test_intraday_ist_to_utc_is_exact_and_no_overnight_or_weekend_bars(self):
        friday, monday = ist_session("2025-01-10"), ist_session("2025-01-13")
        gaps = pd.DatetimeIndex([
            pd.Timestamp("2025-01-10 10:00", tz=IST),
            pd.Timestamp("2025-01-13 11:30", tz=IST),
        ])
        raw = bars(friday.append(monday).difference(gaps))

        out = run(raw, calendar_tz=IST)

        # Exact offsets: 09:15 IST == 03:45 UTC, 15:29 IST == 09:59 UTC.
        assert out.index[0] == pd.Timestamp("2025-01-10 03:45", tz="UTC")
        assert out.index[len(friday) - 1] == pd.Timestamp("2025-01-10 09:59", tz="UTC")
        # Synthetic bars land on the exact UTC instants of the missing minutes.
        added = synthetic_rows(out, raw)
        assert added.equals(gaps.tz_convert("UTC"))
        assert added[0] == pd.Timestamp("2025-01-10 04:30", tz="UTC")
        # Nothing invented between Friday close and Monday open.
        fri_close, mon_open = friday[-1].tz_convert("UTC"), monday[0].tz_convert("UTC")
        assert out[(out.index > fri_close) & (out.index < mon_open)].empty
        assert len(out) == len(friday) + len(monday)
        # Every bar sits inside a session when viewed in IST.
        local = out.index.tz_convert(IST)
        assert (local.day_of_week < 5).all()
        assert ((local.hour * 60 + local.minute >= 9 * 60 + 15) & (local.hour * 60 + local.minute <= 15 * 60 + 29)).all()

    def test_end_to_end_through_fyers_provider_with_epoch_candles(self):
        """
        The real FyersDataSource (calendar_tz=IST, cache on) fed epoch-second
        daily candles with one missing weekday: output has the filled weekday
        and zero weekend candles. Cache is isolated by the conftest fixture.
        """
        fymod = pytest.importorskip("core.data.fyers")
        sessions = pd.bdate_range("2025-01-06", "2025-01-24", tz=IST)
        missing_wed = pd.Timestamp("2025-01-15", tz=IST)

        class FakeClient:
            def history(self, payload):
                return {
                    "candles": [
                        [int(ts.timestamp()), 10.0, 11.0, 9.0, 10.5, 1000]
                        for ts in sessions.drop(missing_wed)
                    ]
                }

        src = fymod.FyersDataSource(fyers_client=FakeClient())
        out = src.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-24", interval="D")

        local = out.index.tz_convert(IST)
        assert local.equals(sessions)                       # all 15 sessions, nothing else
        assert (local.day_of_week < 5).all()                    # no Saturday/Sunday candles
        filled = out.loc[missing_wed.tz_convert("UTC")]
        assert filled["Volume"] == 0
        assert filled["Open"] == filled["High"] == filled["Low"] == filled["Close"] == 10.5
        assert (out.drop(missing_wed.tz_convert("UTC"))["Volume"] == 1000).all()