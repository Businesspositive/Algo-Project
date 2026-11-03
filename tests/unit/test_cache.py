"""
Tests for the local caching layer in core/data/base.py (DataCache + the
cache-aware path of DataSource.fetch).

Core guarantees:
1. A cache miss results in exactly one live provider call.
2. A repeat of the same request results in zero live provider calls.
3. A request inside the cached date range is served as a slice, no call.
4. A request extending past the cached tail fetches only the missing delta.

Plus the live-edge (TTL) rules, integrity rules (synthetic bars never
persisted, empty/failed fetches never claim coverage) and the SQLite store.
"""
import sqlite3
import threading

import numpy as np
import pandas as pd
import pytest

from core.data.base import DEFAULT_CACHE, DataCache, DataSource, OHLCV_COLUMNS

IST = "Asia/Kolkata"
D = "%Y-%m-%d"


# ---------------------------------------------------------------------------
# Test double
# ---------------------------------------------------------------------------
def bars_for(days: pd.DatetimeIndex) -> pd.DataFrame:
    """Closes are a pure function of the date, so any merge can be checked
    against a direct fetch of the same range."""
    close = days.day_of_year.to_numpy(dtype=float) + days.year.to_numpy() % 100
    return pd.DataFrame(
        {
            "Open": close - 0.25,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": (days.day_of_year * 10).to_numpy(dtype="int64"),
        },
        index=days.tz_convert("UTC"),
    )


class RecordingProvider(DataSource):
    """
    Fyers-like provider: daily bars stamped 00:00 IST (18:30 UTC the day
    before), inclusive ``end``, Monday-Friday only, ``end=None`` == today.
    Records every live call and can be made to fail or skip days.
    """

    calendar_tz = IST

    def __init__(self, cache_dir, *, skip=(), fail_after: int | None = None):
        self.cache = DataCache(cache_dir)
        self.calls: list[tuple[str, str | None]] = []
        self.skip = {pd.Timestamp(d, tz=IST) for d in skip}
        self.fail_after = fail_after

    def _fetch_raw(self, symbol, start, end=None, interval="D"):
        self.calls.append((start, end))
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError("simulated provider outage / 429")
        end = end or pd.Timestamp.now(tz=IST).strftime(D)
        days = pd.bdate_range(start, end, tz=IST)
        days = days[~days.isin(self.skip)]
        return bars_for(days) if len(days) else self.empty_frame()


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "cache"


@pytest.fixture
def provider(cache_dir):
    return RecordingProvider(cache_dir)


def direct(start: str, end: str) -> pd.DataFrame:
    """What an uncached provider call for [start, end] returns, standardised."""
    df = DataSource.standardise(bars_for(pd.bdate_range(start, end, tz=IST)))
    df.index.freq = None  # served frames carry no freq metadata
    return df


def ist_dates(df: pd.DataFrame) -> list:
    return [t.date() for t in df.index.tz_convert(IST)]


def set_fetched_at(provider: RecordingProvider, symbol: str, interval: str, when: pd.Timestamp) -> None:
    """Back-date the entry's last provider call through the public API."""
    entry = provider.cache.load(type(provider).__name__, symbol, interval)
    provider.cache.write(
        type(provider).__name__, symbol, interval, entry.df,
        entry.covered_start, entry.covered_end, fetched_at=when, replace=True,
    )


# ===========================================================================
# 1-4. Miss / hit / slice / delta
# ===========================================================================
class TestMissHitSliceDelta:
    def test_cache_miss_results_in_exactly_one_live_call(self, provider, cache_dir):
        out = provider.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-17", "D")

        assert provider.calls == [("2025-01-06", "2025-01-17")]
        pd.testing.assert_frame_equal(out, direct("2025-01-06", "2025-01-17"))
        entry = provider.cache.load("RecordingProvider", "NSE:SBIN-EQ", "D")
        assert entry is not None
        assert (entry.covered_start.strftime(D), entry.covered_end.strftime(D)) == ("2025-01-06", "2025-01-17")
        assert (cache_dir / "market_data.sqlite").exists()

    def test_cache_hit_with_same_parameters_makes_zero_live_calls(self, provider):
        first = provider.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-17", "D")
        provider.calls.clear()

        second = provider.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-17", "D")

        assert provider.calls == []
        pd.testing.assert_frame_equal(second, first)

    def test_hit_survives_a_new_provider_instance(self, cache_dir):
        RecordingProvider(cache_dir).fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-17", "D")
        fresh = RecordingProvider(cache_dir)  # e.g. the next process / next run

        out = fresh.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-17", "D")

        assert fresh.calls == []
        pd.testing.assert_frame_equal(out, direct("2025-01-06", "2025-01-17"))

    def test_request_inside_cached_range_is_served_as_slice_without_network(self, provider):
        provider.fetch("NSE:SBIN-EQ", "2025-01-01", "2025-01-31", "D")
        provider.calls.clear()

        out = provider.fetch("NSE:SBIN-EQ", "2025-01-13", "2025-01-17", "D")

        assert provider.calls == []
        # Exactly Mon 13 .. Fri 17 in IST -- including Monday's bar, which is
        # stamped Sunday 18:30 UTC and would be lost by a naive UTC slice.
        assert ist_dates(out) == [pd.Timestamp(d).date() for d in
                                  ["2025-01-13", "2025-01-14", "2025-01-15", "2025-01-16", "2025-01-17"]]
        assert out.index[0] == pd.Timestamp("2025-01-12 18:30", tz="UTC")
        pd.testing.assert_frame_equal(out, direct("2025-01-13", "2025-01-17"))

    def test_slice_starting_on_a_weekend_is_still_a_hit(self, provider):
        provider.fetch("NSE:SBIN-EQ", "2025-01-01", "2025-01-31", "D")
        provider.calls.clear()

        out = provider.fetch("NSE:SBIN-EQ", "2025-01-11", "2025-01-12", "D")  # Sat..Sun

        assert provider.calls == []
        assert out.empty
        assert list(out.columns) == list(OHLCV_COLUMNS)

    def test_extending_past_cached_tail_fetches_only_the_delta(self, provider):
        provider.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-17", "D")
        provider.calls.clear()

        out = provider.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-31", "D")

        # One call, from the old coverage end (one-day seam overlap) to the new end.
        assert provider.calls == [("2025-01-17", "2025-01-31")]
        # Merged result is indistinguishable from a direct fetch of the whole range.
        pd.testing.assert_frame_equal(out, direct("2025-01-06", "2025-01-31"))
        entry = provider.cache.load("RecordingProvider", "NSE:SBIN-EQ", "D")
        assert entry.covered_end.strftime(D) == "2025-01-31"
        assert not entry.df.index.has_duplicates

    def test_daily_loop_pattern_fetches_one_small_delta_per_day(self, provider):
        """DailySelector: fixed start, end moves forward one day per run."""
        provider.fetch("NSE:SBIN-EQ", "2025-01-01", "2025-01-20", "D")
        provider.calls.clear()
        for day in ("2025-01-21", "2025-01-22", "2025-01-23"):
            provider.fetch("NSE:SBIN-EQ", "2025-01-01", day, "D")

        assert provider.calls == [
            ("2025-01-20", "2025-01-21"),
            ("2025-01-21", "2025-01-22"),
            ("2025-01-22", "2025-01-23"),
        ]

    def test_extending_before_cached_head_fetches_only_the_delta(self, provider):
        provider.fetch("NSE:SBIN-EQ", "2025-01-13", "2025-01-31", "D")
        provider.calls.clear()

        out = provider.fetch("NSE:SBIN-EQ", "2025-01-01", "2025-01-31", "D")

        assert provider.calls == [("2025-01-01", "2025-01-13")]
        pd.testing.assert_frame_equal(out, direct("2025-01-01", "2025-01-31"))

    def test_rolling_window_fetches_tail_only(self, provider):
        """Orchestrator: both start and end slide forward; start is still covered."""
        provider.fetch("NSE:SBIN-EQ", "2025-01-01", "2025-01-31", "D")
        provider.calls.clear()

        out = provider.fetch("NSE:SBIN-EQ", "2025-01-02", "2025-02-03", "D")

        assert provider.calls == [("2025-01-31", "2025-02-03")]
        pd.testing.assert_frame_equal(out, direct("2025-01-02", "2025-02-03"))

    def test_disjoint_later_request_bridges_the_gap_and_keeps_coverage_contiguous(self, provider):
        provider.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-10", "D")
        provider.calls.clear()

        out = provider.fetch("NSE:SBIN-EQ", "2025-02-10", "2025-02-14", "D")

        assert provider.calls == [("2025-01-10", "2025-02-14")]
        pd.testing.assert_frame_equal(out, direct("2025-02-10", "2025-02-14"))
        entry = provider.cache.load("RecordingProvider", "NSE:SBIN-EQ", "D")
        assert (entry.covered_start.strftime(D), entry.covered_end.strftime(D)) == ("2025-01-06", "2025-02-14")
        # Everything in between is now a hit.
        provider.calls.clear()
        provider.fetch("NSE:SBIN-EQ", "2025-01-20", "2025-01-24", "D")
        assert provider.calls == []

    def test_entries_are_keyed_by_provider_symbol_and_interval(self, cache_dir):
        a = RecordingProvider(cache_dir)
        a.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-10", "D")

        class OtherProvider(RecordingProvider):
            pass

        b = OtherProvider(cache_dir)
        b.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-10", "D")           # other provider
        a.fetch("NSE:TCS-EQ", "2025-01-06", "2025-01-10", "D")            # other symbol
        a.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-10", "60")          # other interval

        assert len(b.calls) == 1
        assert len(a.calls) == 3

    def test_cache_disabled_always_calls_provider(self, provider):
        provider.cache = None
        provider.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-10", "D")
        provider.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-10", "D")
        assert len(provider.calls) == 2

    def test_default_cache_location_follows_environment_variable(self, tmp_path, monkeypatch):
        class ApiProvider(RecordingProvider):
            cache = DEFAULT_CACHE

            def __init__(self):
                self.calls, self.skip, self.fail_after = [], set(), None

        target = tmp_path / "relocated"
        monkeypatch.setenv("MARKET_DATA_CACHE_DIR", str(target))
        src = ApiProvider()
        src.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-10", "D")
        src.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-10", "D")

        assert (target / "market_data.sqlite").exists()
        assert len(src.calls) == 1


# ===========================================================================
# Live edge (today's bars are provisional)
# ===========================================================================
class TestLiveEdge:
    @pytest.fixture
    def today(self):
        return pd.Timestamp.now(tz=IST).normalize().tz_localize(None)

    def test_same_day_request_within_ttl_is_a_hit(self, provider, today):
        start = (today - pd.Timedelta(days=10)).strftime(D)
        provider.fetch("NSE:SBIN-EQ", start, None, "D")
        provider.calls.clear()

        provider.fetch("NSE:SBIN-EQ", start, None, "D")

        assert provider.calls == []

    def test_same_day_request_after_ttl_refetches_only_the_provisional_day(self, provider, today):
        start = (today - pd.Timedelta(days=10)).strftime(D)
        provider.fetch("NSE:SBIN-EQ", start, None, "D")
        set_fetched_at(provider, "NSE:SBIN-EQ", "D", pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2))
        provider.calls.clear()

        provider.fetch("NSE:SBIN-EQ", start, None, "D")

        # Tail from yesterday (seam overlap) to "now"; end=None is preserved.
        assert provider.calls == [((today - pd.Timedelta(days=1)).strftime(D), None)]

    def test_entry_fetched_yesterday_refetches_yesterday_and_today(self, provider, today):
        start = (today - pd.Timedelta(days=10)).strftime(D)
        provider.fetch("NSE:SBIN-EQ", start, None, "D")
        set_fetched_at(provider, "NSE:SBIN-EQ", "D", pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1))
        provider.calls.clear()

        provider.fetch("NSE:SBIN-EQ", start, None, "D")

        assert provider.calls == [((today - pd.Timedelta(days=2)).strftime(D), None)]

    def test_historical_range_is_never_provisional(self, provider):
        provider.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-17", "D")
        set_fetched_at(provider, "NSE:SBIN-EQ", "D", pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30))
        provider.calls.clear()

        provider.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-17", "D")

        assert provider.calls == []

    def test_refresh_bypasses_cache_and_replaces_entry(self, provider):
        provider.fetch("NSE:SBIN-EQ", "2025-01-01", "2025-01-31", "D")
        provider.calls.clear()

        provider.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-10", "D", refresh=True)

        assert provider.calls == [("2025-01-06", "2025-01-10")]
        entry = provider.cache.load("RecordingProvider", "NSE:SBIN-EQ", "D")
        assert (entry.covered_start.strftime(D), entry.covered_end.strftime(D)) == ("2025-01-06", "2025-01-10")
        assert len(entry.df) == 5


# ===========================================================================
# Integrity
# ===========================================================================
class TestIntegrity:
    def test_gap_filled_synthetic_bars_are_served_but_never_persisted(self, cache_dir):
        src = RecordingProvider(cache_dir, skip=["2025-01-08"])
        out = src.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-10", "D")
        wed = pd.Timestamp("2025-01-08", tz=IST).tz_convert("UTC")

        assert wed in out.index and out.loc[wed, "Volume"] == 0        # served, flat, zero volume
        on_disk = src.cache.load("RecordingProvider", "NSE:SBIN-EQ", "D").df
        assert wed not in on_disk.index                                   # not persisted
        assert len(on_disk) == 4

    def test_empty_provider_result_is_not_cached(self, provider):
        provider.fetch("NSE:SBIN-EQ", "2025-01-11", "2025-01-12", "D")  # weekend only -> empty
        provider.fetch("NSE:SBIN-EQ", "2025-01-11", "2025-01-12", "D")

        assert len(provider.calls) == 2
        assert provider.cache.load("RecordingProvider", "NSE:SBIN-EQ", "D") is None

    def test_provider_failure_during_tail_extension_leaves_cache_untouched(self, cache_dir):
        src = RecordingProvider(cache_dir, fail_after=1)
        src.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-17", "D")
        before = src.cache.load("RecordingProvider", "NSE:SBIN-EQ", "D")

        with pytest.raises(RuntimeError, match="429"):
            src.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-31", "D")

        after = src.cache.load("RecordingProvider", "NSE:SBIN-EQ", "D")
        assert after.covered_end == before.covered_end == pd.Timestamp("2025-01-17")
        pd.testing.assert_frame_equal(after.df, before.df)
        # And the failure is not "remembered": the delta is retried next time.
        src.fail_after = None
        src.calls.clear()
        src.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-31", "D")
        assert src.calls == [("2025-01-17", "2025-01-31")]

    def test_fyers_rate_limit_response_raises_instead_of_extending_coverage(self, cache_dir):
        fymod = pytest.importorskip("core.data.fyers")

        class Client:
            def __init__(self):
                self.n = 0

            def history(self, payload):
                self.n += 1
                if self.n > 1:
                    return {"s": "error", "code": 429, "message": "request limit reached"}
                days = pd.bdate_range(payload["range_from"], payload["range_to"], tz=IST)
                return {"s": "ok", "candles": [[int(t.timestamp()), 1.0, 2.0, 0.5, 1.5, 10] for t in days]}

        src = fymod.FyersDataSource(fyers_client=Client())
        src.cache = DataCache(cache_dir)
        src.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-17", interval="D")

        with pytest.raises(RuntimeError, match="request limit reached"):
            src.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-31", interval="D")

        entry = src.cache.load("FyersDataSource", "NSE:SBIN-EQ", "D")
        assert entry.covered_end == pd.Timestamp("2025-01-17")  # the failed days are still owed

    def test_served_frames_pass_the_format_contract(self, provider):
        provider.fetch("NSE:SBIN-EQ", "2025-01-06", "2025-01-17", "D")
        out = provider.fetch("NSE:SBIN-EQ", "2025-01-08", "2025-01-31", "D")  # slice + tail, through cache
        pd.testing.assert_frame_equal(DataSource.standardise(out), out)
        assert out["Volume"].dtype == "int64"

    def test_invalid_dates_raise_clear_value_error(self, provider):
        with pytest.raises(ValueError, match="not a valid date"):
            provider.fetch("NSE:SBIN-EQ", "not-a-date", "2025-01-10", "D")
        with pytest.raises(ValueError, match="is after end"):
            provider.fetch("NSE:SBIN-EQ", "2025-01-10", "2025-01-06", "D")

    def test_tz_aware_bounds_are_interpreted_in_calendar_tz(self, provider):
        out = provider.fetch("NSE:SBIN-EQ", "2025-01-06T00:00:00+05:30", "2025-01-10T23:59:00+05:30", "D")
        assert ist_dates(out)[0] == pd.Timestamp("2025-01-06").date()
        assert len(out) == 5


# ===========================================================================
# DataCache (SQLite store)
# ===========================================================================
class TestDataCacheStore:
    @pytest.fixture
    def store(self, cache_dir):
        return DataCache(cache_dir)

    @staticmethod
    def frame(n=50, start="2025-01-01 03:45", freq="1min"):
        """A standardised frame -- the only kind the store ever receives."""
        idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
        rng = np.random.default_rng(0)
        raw = pd.DataFrame(
            {
                "Open": rng.random(n) * 1e4, "High": rng.random(n) * 1e4,
                "Low": rng.random(n) * 1e4, "Close": rng.random(n) * 1e4,
                "Volume": rng.integers(0, 5_000_000_000, n, dtype="int64"),
            },
            index=idx,
        )
        return DataSource.standardise(raw)

    def test_round_trip_is_exact_including_dtypes_and_timezone(self, store):
        df = self.frame()
        now = pd.Timestamp.now(tz="UTC")
        store.write("P", "S", "1", df, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02"), fetched_at=now)

        entry = store.load("P", "S", "1")

        pd.testing.assert_frame_equal(entry.df, df, check_freq=False)
        assert entry.df["Volume"].dtype == "int64"
        assert str(entry.df.index.tz) == "UTC"
        assert entry.fetched_at == now
        assert (entry.covered_start, entry.covered_end) == (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02"))

    def test_windowed_load_reads_only_the_requested_range(self, store):
        df = self.frame(100)
        store.write("P", "S", "1", df, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02"), fetched_at=pd.Timestamp.now(tz="UTC"))
        lo, hi = df.index[10], df.index[20]

        entry = store.load("P", "S", "1", window=(lo, hi))

        pd.testing.assert_frame_equal(entry.df, df.iloc[10:20], check_freq=False)   # lo inclusive, hi exclusive
        assert entry.covered_end == pd.Timestamp("2025-01-02")                        # metadata still complete

    def test_delta_write_upserts_new_bars_and_replaces_overlapping_ones(self, store):
        base = self.frame(10)
        now = pd.Timestamp.now(tz="UTC")
        store.write("P", "S", "1", base, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01"), fetched_at=now)
        delta = self.frame(5, start=str(base.index[8].tz_localize(None)))  # overlaps last 2 bars, adds 3
        delta = delta.copy()
        delta[:] = -1.0
        delta["Volume"] = 7

        store.write("P", "S", "1", delta, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02"), fetched_at=now)

        out = store.load("P", "S", "1").df
        assert len(out) == 13
        assert (out.loc[delta.index, "Close"] == -1.0).all()                 # newer rows win at the seam
        pd.testing.assert_frame_equal(out.iloc[:8], base.iloc[:8], check_freq=False)

    def test_drop_from_removes_provisional_bars_in_the_same_transaction(self, store):
        df = self.frame(10)
        now = pd.Timestamp.now(tz="UTC")
        store.write("P", "S", "1", df, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01"), fetched_at=now)

        store.write("P", "S", "1", df.iloc[:0], pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01"),
                    fetched_at=now, drop_from=df.index[6])

        out = store.load("P", "S", "1").df
        pd.testing.assert_frame_equal(out, df.iloc[:6], check_freq=False)

    def test_replace_discards_previous_bars(self, store):
        now = pd.Timestamp.now(tz="UTC")
        store.write("P", "S", "1", self.frame(10), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01"), fetched_at=now)
        small = self.frame(3, start="2025-06-01")
        store.write("P", "S", "1", small, pd.Timestamp("2025-06-01"), pd.Timestamp("2025-06-01"), fetched_at=now, replace=True)
        pd.testing.assert_frame_equal(store.load("P", "S", "1").df, small, check_freq=False)

    def test_nan_round_trips(self, store):
        df = self.frame(3)
        df.iloc[1, df.columns.get_loc("High")] = np.nan
        store.write("P", "S", "1", df, pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01"), fetched_at=pd.Timestamp.now(tz="UTC"))
        assert store.load("P", "S", "1").df["High"].isna().tolist() == [False, True, False]

    def test_clear_filters(self, store):
        now = pd.Timestamp.now(tz="UTC")
        for key in [("A", "X", "1"), ("A", "Y", "1"), ("A", "X", "D"), ("B", "X", "1")]:
            store.write(*key, self.frame(3), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01"), fetched_at=now)

        assert store.clear(provider="A", symbol="X", interval="1") == 1
        assert store.clear(symbol="Y") == 1
        assert store.clear(provider="B") == 1
        assert store.load("A", "X", "D") is not None
        assert store.clear() == 1
        with sqlite3.connect(store.path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0] == 0

    def test_corrupt_metadata_is_a_miss_with_a_warning(self, store, caplog):
        store.write("P", "S", "1", self.frame(3), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01"), fetched_at=pd.Timestamp.now(tz="UTC"))
        with sqlite3.connect(store.path) as conn:
            conn.execute("UPDATE series SET dtypes = 'not json'")

        with caplog.at_level("WARNING", logger="core.data.base"):
            assert store.load("P", "S", "1") is None
        assert "unreadable" in caplog.text

    def test_schema_version_mismatch_is_a_miss(self, store):
        store.write("P", "S", "1", self.frame(3), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01"), fetched_at=pd.Timestamp.now(tz="UTC"))
        with sqlite3.connect(store.path) as conn:
            conn.execute("UPDATE series SET version = version + 1")
        assert store.load("P", "S", "1") is None

    def test_missing_key_is_none_and_clear_on_empty_store_is_zero(self, store):
        assert store.load("P", "S", "1") is None
        assert store.clear() == 0

    def test_explicit_database_file_path_is_honoured(self, tmp_path):
        store = DataCache(tmp_path / "nested" / "quotes.db")
        store.write("P", "S", "1", self.frame(2), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01"), fetched_at=pd.Timestamp.now(tz="UTC"))
        assert (tmp_path / "nested" / "quotes.db").exists()

    def test_concurrent_writers_are_serialised_without_loss(self, store):
        now = pd.Timestamp.now(tz="UTC")
        errors = []

        def worker(i):
            try:
                for _ in range(5):
                    store.write("P", f"S{i}", "1", self.frame(20), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01"), fetched_at=now)
            except Exception as exc:  # pragma: no cover - only on failure
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        for i in range(6):
            assert len(store.load("P", f"S{i}", "1").df) == 20