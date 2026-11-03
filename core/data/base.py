from __future__ import annotations

import json
import logging
import os
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

#: Canonical OHLCV schema. Every frame handed out by a ``DataSource`` has
#: exactly these columns, in this order.
OHLCV_COLUMNS: Tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")

_ONE_DAY = pd.Timedelta(days=1)

def default_cache_dir() -> Path:
    """
    Directory for the on-disk market-data cache. Resolved at call time (not
    import time) so ``MARKET_DATA_CACHE_DIR`` can be set by the environment,
    a container, or a test fixture. Defaults to ``<project>/.cache/market_data``.
    """
    env = os.environ.get("MARKET_DATA_CACHE_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[2] / ".cache" / "market_data"


class _DefaultCache:
    """Sentinel meaning 'use the default cache location' (see ``default_cache_dir``)."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "DEFAULT_CACHE"


#: Assign to ``DataSource.cache`` to enable caching at the default location.
DEFAULT_CACHE = _DefaultCache()


@dataclass(frozen=True)
class CacheEntry:
    """One cached series plus the request range it is known to satisfy."""

    df: pd.DataFrame
    covered_start: pd.Timestamp  # naive calendar date (inclusive)
    covered_end: pd.Timestamp    # naive calendar date (inclusive)
    fetched_at: pd.Timestamp     # tz-aware UTC, time of the most recent provider call


class DataCache:
    """
    SQLite-backed store of *standardised* OHLCV frames.

    One entry per ``(provider, symbol, interval)``: a row in ``series`` with
    the covered date range, last provider-call time and column dtypes, and
    the bars themselves in ``bars`` keyed by epoch-nanosecond UTC timestamp.
    Reads can be windowed to a time range and writes are *deltas* (upsert of
    new bars, optional trim of provisional ones, metadata update) applied in
    a single ``BEGIN IMMEDIATE`` transaction, so cost scales with the request
    rather than with the symbol's history, readers in other processes never
    observe a half-written series, and concurrent writers are serialised by
    SQLite itself. Frames are re-validated through ``DataSource.standardise``
    on load; unreadable content is logged and treated as a miss.

    ``path`` may be a directory (the database is ``<dir>/market_data.sqlite``)
    or an explicit ``.sqlite`` / ``.db`` file.
    """

    VERSION = 2
    DB_SUFFIXES = (".sqlite", ".sqlite3", ".db")
    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS series (
            provider      TEXT    NOT NULL,
            symbol        TEXT    NOT NULL,
            interval      TEXT    NOT NULL,
            version       INTEGER NOT NULL,
            covered_start TEXT    NOT NULL,
            covered_end   TEXT    NOT NULL,
            fetched_at    TEXT    NOT NULL,
            dtypes        TEXT    NOT NULL,
            rows          INTEGER NOT NULL,
            PRIMARY KEY (provider, symbol, interval)
        );
        CREATE TABLE IF NOT EXISTS bars (
            provider TEXT    NOT NULL,
            symbol   TEXT    NOT NULL,
            interval TEXT    NOT NULL,
            ts       INTEGER NOT NULL,
            open     REAL,
            high     REAL,
            low      REAL,
            close    REAL,
            volume   REAL,
            PRIMARY KEY (provider, symbol, interval, ts)
        ) WITHOUT ROWID;
    """
    _KEY = "provider = ? AND symbol = ? AND interval = ?"

    def __init__(self, path: Union[str, Path, None] = None):
        path = Path(path) if path is not None else default_cache_dir()
        self.path = path if path.suffix.lower() in self.DB_SUFFIXES else path / "market_data.sqlite"

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(self._SCHEMA)
        return conn

    # --------------------------------------------------------------- I/O
    def load(
        self,
        provider: str,
        symbol: str,
        interval: str,
        window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
    ) -> Optional[CacheEntry]:
        """
        Entry for the key, or ``None`` if absent/unreadable. ``window`` is an
        optional ``(lo, hi)`` pair of tz-aware timestamps; only bars with
        ``lo <= ts < hi`` are read (coverage metadata is always complete).
        """
        key = (provider, symbol, interval)
        try:
            conn = self._connect()
            try:
                row = conn.execute(
                    f"SELECT version, covered_start, covered_end, fetched_at, dtypes "
                    f"FROM series WHERE {self._KEY}", key,
                ).fetchone()
                if row is None:
                    return None
                version, covered_start, covered_end, fetched_at, dtypes = row
                if version != self.VERSION:
                    logger.info("Cache %s %s: version %s != %s, ignoring.", self.path, key, version, self.VERSION)
                    return None
                sql = f"SELECT ts, open, high, low, close, volume FROM bars WHERE {self._KEY}"
                params: Tuple = key
                if window is not None:
                    lo, hi = window
                    sql += " AND ts >= ? AND ts < ?"
                    params = (*key, self._ns(lo), self._ns(hi))
                bars = pd.read_sql_query(sql + " ORDER BY ts", conn, params=params)
            finally:
                conn.close()

            index = pd.to_datetime(bars["ts"].astype("int64"), unit="ns", utc=True)
            index.name = None
            df = pd.DataFrame({col: bars[col.lower()].to_numpy() for col in OHLCV_COLUMNS}, index=index)
            for col, dtype in json.loads(dtypes).items():
                if col in df.columns:
                    try:
                        df[col] = df[col].astype(dtype)
                    except (TypeError, ValueError):  # e.g. NaN into int64: keep float
                        pass
            df = DataSource.standardise(df, context=f"cache {self.path} {key}")
            return CacheEntry(
                df=df,
                covered_start=pd.Timestamp(covered_start),
                covered_end=pd.Timestamp(covered_end),
                fetched_at=pd.Timestamp(fetched_at).tz_convert("UTC"),
            )
        except Exception as exc:  # corrupt entry -> miss, not a crash
            logger.warning("Cache %s %s unreadable (%s); ignoring it.", self.path, key, exc)
            return None

    @staticmethod
    def _ns(ts: pd.Timestamp) -> int:
        """Epoch nanoseconds of a tz-aware timestamp (``.value`` is always ns, whatever the unit)."""
        return int(pd.Timestamp(ts).tz_convert("UTC").as_unit("ns").value)

    def write(
        self,
        provider: str,
        symbol: str,
        interval: str,
        bars: pd.DataFrame,
        covered_start: pd.Timestamp,
        covered_end: pd.Timestamp,
        fetched_at: pd.Timestamp,
        *,
        drop_from: Optional[pd.Timestamp] = None,
        replace: bool = False,
    ) -> None:
        """
        Apply a delta to the entry in one transaction: optionally drop all
        stored bars (``replace``) or those at/after ``drop_from`` (provisional
        trim), upsert ``bars`` (new timestamps win), and record the new
        coverage / ``fetched_at``. ``bars`` may be empty (e.g. a tail fetch
        over a weekend) -- coverage is still extended.
        """
        key = (provider, symbol, interval)
        # .tolist() yields plain Python scalars (sqlite3 cannot bind numpy types).
        rows = zip(
            bars.index.as_unit("ns").asi8.tolist(),  # epoch-ns regardless of index resolution
            *(bars[col].astype("float64").tolist() for col in OHLCV_COLUMNS),
        )
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                existing = conn.execute(f"SELECT dtypes FROM series WHERE {self._KEY}", key).fetchone()
                if replace:
                    conn.execute(f"DELETE FROM bars WHERE {self._KEY}", key)
                elif drop_from is not None:
                    conn.execute(f"DELETE FROM bars WHERE {self._KEY} AND ts >= ?", (*key, self._ns(drop_from)))
                conn.executemany(
                    "INSERT OR REPLACE INTO bars (provider, symbol, interval, ts, open, high, low, close, volume) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    ((*key, *row) for row in rows),
                )
                if len(bars):
                    dtypes = json.dumps({col: str(bars[col].dtype) for col in OHLCV_COLUMNS})
                elif existing is not None and not replace:
                    dtypes = existing[0]
                else:
                    dtypes = json.dumps({col: "float64" for col in OHLCV_COLUMNS})
                n_rows = conn.execute(f"SELECT COUNT(*) FROM bars WHERE {self._KEY}", key).fetchone()[0]
                conn.execute(
                    "INSERT OR REPLACE INTO series "
                    "(provider, symbol, interval, version, covered_start, covered_end, fetched_at, dtypes, rows) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        *key,
                        self.VERSION,
                        covered_start.strftime("%Y-%m-%d"),
                        covered_end.strftime("%Y-%m-%d"),
                        fetched_at.isoformat(),
                        dtypes,
                        int(n_rows),
                    ),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        finally:
            conn.close()

    def clear(
        self,
        provider: Optional[str] = None,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
    ) -> int:
        """Delete matching entries (all of them when no filter is given). Returns count removed."""
        if not self.path.exists():
            return 0
        clauses, params = [], []
        for column, value in (("provider", provider), ("symbol", symbol), ("interval", interval)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            removed = conn.execute(f"DELETE FROM series{where}", params).rowcount
            conn.execute(f"DELETE FROM bars{where}", params)
            conn.execute("COMMIT")
        finally:
            conn.close()
        return removed


class DataSource(ABC):
    """
    Contract every market data provider must satisfy.

    ``fetch`` is a template method built around the provider-specific
    ``_fetch_raw`` hook. Data flows through three stages:

    1. ``standardise`` -- format contract. Every consumer is guaranteed a frame
       that has exactly the columns ``Open, High, Low, Close, Volume`` and is
       indexed by a sorted, unique, timezone-aware ``DatetimeIndex`` in UTC.
       Anything that cannot be brought into this shape raises ``ValueError``
       naming the provider, the symbol and the specific violation.
    2. local cache -- standardised bars are stored on disk per
       ``(provider, symbol, interval)`` together with the date range they are
       known to cover. A request inside that range is served without touching
       the provider; a request that extends it fetches *only* the missing
       head/tail and merges. Bars from the day of the last provider call are
       treated as provisional (the session may not have closed) and are
       re-fetched once ``cache_ttl`` has elapsed. Controlled by the ``cache``
       and ``cache_ttl`` attributes and the ``refresh`` argument of ``fetch``.
    3. ``clean`` -- data-quality pass on the served slice. Missing bars
       (illiquid tickers that skip minutes/days) are detected and filled with
       flat bars: ``Close`` forward-filled, ``Open/High/Low`` set to that close
       and ``Volume`` set to zero. Observed bars are never altered or dropped.
       Runs *after* the cache so synthetic bars are never persisted. Tunable
       via ``fill_gaps`` / ``session_gap`` / ``calendar_tz`` and the
       ``trading_days`` hook.

    Subclasses must implement ``_fetch_raw`` and must *not* override ``fetch``
    (doing so would bypass the validation; it is rejected at class-creation
    time). Providers that have nothing to return should use ``empty_frame()``
    rather than a bare ``pd.DataFrame()`` so the schema holds even when empty.
    """

    #: Run the gap-filling ``clean`` stage inside ``fetch``.
    fill_gaps: bool = True

    #: Intraday bars only: a gap longer than ``max(session_gap, 2 * bar_size)``
    #: is treated as a session boundary (overnight, weekend, holiday) and left
    #: alone; anything shorter is assumed to be missing bars and is filled.
    session_gap: pd.Timedelta = pd.Timedelta(hours=4)

    #: Timezone the exchange calendar is defined in. Daily bars are stamped by
    #: most providers at local midnight, which in UTC can fall on the previous
    #: calendar day -- weekday checks therefore happen in this zone.
    calendar_tz: str = "UTC"

    #: Where standardised bars are cached: ``DEFAULT_CACHE`` (the location
    #: given by ``default_cache_dir()`` / ``MARKET_DATA_CACHE_DIR``), a
    #: ``DataCache``, a directory or ``.sqlite`` path, or ``None`` for no
    #: caching. Caching is only correct when the data is a pure function of
    #: ``(symbol, interval, date range)`` -- i.e. remote APIs -- so API-backed
    #: providers opt in with ``cache = DEFAULT_CACHE`` while in-memory, replay
    #: and test sources keep the ``None`` default.
    cache: Union[_DefaultCache, "DataCache", str, Path, None] = None

    #: How long a cache entry that reaches the current day may be served
    #: before its provisional (same-day) bars are re-fetched from the provider.
    cache_ttl: pd.Timedelta = pd.Timedelta(minutes=15)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "fetch" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not override DataSource.fetch(); "
                "implement _fetch_raw() instead so the OHLCV/UTC contract is enforced."
            )

    # ------------------------------------------------------------------ API
    def fetch(
        self,
        symbol: str,
        start: str,
        end: Optional[str] = None,
        interval: str = "1d",
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return an OHLCV DataFrame for ``symbol`` with exactly the columns
        ``['Open','High','Low','Close','Volume']`` and a tz-aware UTC
        ``DatetimeIndex``, served from the local cache where possible and with
        missing bars filled (see ``clean``) unless ``fill_gaps`` is False.

        ``start``/``end`` are ``YYYY-MM-DD`` calendar dates; ``end=None`` means
        today. ``refresh=True`` bypasses the cache read and replaces the cached
        entry with a fresh provider fetch.

        Raises
        ------
        ValueError
            If the underlying provider returned data that does not match the
            column schema or whose index cannot be standardised to UTC.
        """
        context = f"{type(self).__name__}.fetch({symbol!r})"
        cache = self._resolve_cache()

        if cache is None:
            df = self._fetch_standardised(symbol, start, end, interval, context)
        else:
            df = self._fetch_cached(cache, symbol, start, end, interval, context, refresh=refresh)

        if self.fill_gaps:
            df = self.clean(df, context=context)
        return df

    @abstractmethod
    def _fetch_raw(
        self,
        symbol: str,
        start: str,
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Provider-specific retrieval. Should return a DataFrame with the OHLCV
        columns and a ``DatetimeIndex`` (naive timestamps are treated as UTC).
        Return ``self.empty_frame()`` when there is no data.
        """
        raise NotImplementedError

    # -------------------------------------------------------------- caching
    def _resolve_cache(self) -> Optional[DataCache]:
        cache = self.cache
        if cache is None:
            return None
        if isinstance(cache, DataCache):
            return cache
        if isinstance(cache, _DefaultCache):
            return DataCache(default_cache_dir())
        return DataCache(cache)

    def _fetch_standardised(
        self,
        symbol: str,
        start: str,
        end: Optional[str],
        interval: str,
        context: str,
    ) -> pd.DataFrame:
        """Provider call + format contract, no caching, no gap filling."""
        raw = self._fetch_raw(symbol, start=start, end=end, interval=interval)
        return self.standardise(raw, context=context)

    def _today(self) -> pd.Timestamp:
        """Current calendar date in ``calendar_tz`` as a naive midnight Timestamp."""
        return pd.Timestamp.now(tz=self.calendar_tz).normalize().tz_localize(None)

    def _as_date(self, value, *, name: str, context: str) -> pd.Timestamp:
        """Parse a request bound to a naive calendar date (tz-aware input is read in ``calendar_tz``)."""
        try:
            ts = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}: {name}={value!r} is not a valid date.") from exc
        if ts is pd.NaT:
            raise ValueError(f"{context}: {name}={value!r} is not a valid date.")
        if ts.tz is not None:
            ts = ts.tz_convert(self.calendar_tz).tz_localize(None)
        return ts.normalize()

    def _bounds(self, start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """UTC instants ``[lo, hi)`` spanning calendar days ``start..end`` in ``calendar_tz``."""
        lo = start.tz_localize(self.calendar_tz).tz_convert("UTC")
        hi = (end + _ONE_DAY).tz_localize(self.calendar_tz).tz_convert("UTC")
        return lo, hi

    def _slice(self, df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Rows whose timestamp falls on calendar days ``start..end`` in ``calendar_tz``."""
        lo, hi = self._bounds(start, end)
        return df[(df.index >= lo) & (df.index < hi)]

    def _fetch_cached(
        self,
        cache: DataCache,
        symbol: str,
        start: str,
        end: Optional[str],
        interval: str,
        context: str,
        *,
        refresh: bool,
    ) -> pd.DataFrame:
        """
        Serve ``[start, end]`` from the cache, fetching only what is missing.

        Coverage is tracked as the calendar-date range of requests already
        satisfied, not as the first/last bar (a request starting on a Saturday
        has no bar on its start date but is still fully covered). Only the
        requested window is read from disk. On a partial miss the missing head
        and/or tail is fetched with a one-day overlap at the seam, so providers
        with exclusive ``end`` semantics cannot leave a hole; duplicates are
        resolved in favour of the newer fetch, and only the delta is written.
        Bars dated on/after the day of the previous provider call are
        provisional and are dropped and re-fetched once ``cache_ttl`` expires.
        """
        provider = type(self).__name__
        today = self._today()
        req_start = self._as_date(start, name="start", context=context)
        req_end = self._as_date(end, name="end", context=context) if end else today
        if req_start > req_end:
            raise ValueError(f"{context}: start {start!r} is after end {end!r}.")
        now = pd.Timestamp.now(tz="UTC")
        fmt = "%Y-%m-%d"

        entry = None if refresh else cache.load(provider, symbol, interval, window=self._bounds(req_start, req_end))

        drop_from = None
        if entry is not None:
            cached, cov_start, cov_end = entry.df, entry.covered_start, entry.covered_end
            provisional_from = entry.fetched_at.tz_convert(self.calendar_tz).normalize().tz_localize(None)
            if cov_end >= provisional_from and (now - entry.fetched_at) > self.cache_ttl:
                drop_from = provisional_from.tz_localize(self.calendar_tz).tz_convert("UTC")
                cached = cached[cached.index < drop_from]
                cov_end = provisional_from - _ONE_DAY
                if cov_end < cov_start:
                    entry = None

        if entry is None:
            logger.info("%s: cache miss, fetching %s..%s.", context, req_start.date(), req_end.date())
            new = self._fetch_standardised(
                symbol, req_start.strftime(fmt), None if end is None else req_end.strftime(fmt), interval, context)
            if new.empty:
                # Nothing to claim coverage for. Don't leave a stale entry behind on refresh.
                if refresh:
                    cache.clear(provider, symbol, interval)
                return new
            cache.write(provider, symbol, interval, new, req_start, req_end, fetched_at=now, replace=True)
            return self._slice(new, req_start, req_end)

        if cov_start <= req_start and req_end <= cov_end:
            logger.debug("%s: cache hit (%s..%s).", context, cov_start.date(), cov_end.date())
            return cached  # already windowed to the request by the load

        parts = []
        if req_start < cov_start:
            logger.info("%s: extending cache head %s..%s.", context, req_start.date(), cov_start.date())
            parts.append(self._fetch_standardised(
                symbol, req_start.strftime(fmt), cov_start.strftime(fmt), interval, context))
        if req_end > cov_end:
            # Preserve the caller's ``end=None`` so providers that treat it as
            # "up to now" (rather than an exclusive date) keep doing so.
            tail_end = None if end is None else req_end.strftime(fmt)
            logger.info("%s: extending cache tail %s..%s.", context, cov_end.date(), req_end.date())
            parts.append(self._fetch_standardised(
                symbol, cov_end.strftime(fmt), tail_end, interval, context))
        new = pd.concat(parts)
        new = new[~new.index.duplicated(keep="last")].sort_index()

        cache.write(
            provider, symbol, interval, new,
            min(req_start, cov_start), max(req_end, cov_end),
            fetched_at=now, drop_from=drop_from,
        )
        served = pd.concat([cached, new])
        served = served[~served.index.duplicated(keep="last")].sort_index()
        return self._slice(served, req_start, req_end)

    def clear_cache(self, symbol: Optional[str] = None, interval: Optional[str] = None) -> int:
        """Drop this provider's cached entries (optionally one symbol/interval). Returns count removed."""
        cache = self._resolve_cache()
        if cache is None:
            return 0
        return cache.clear(type(self).__name__, symbol=symbol, interval=interval)

    # -------------------------------------------------------------- helpers
    @staticmethod
    def empty_frame() -> pd.DataFrame:
        """A schema-conformant frame with zero rows."""
        return pd.DataFrame(
            {col: pd.Series(dtype="float64") for col in OHLCV_COLUMNS},
            index=pd.DatetimeIndex([], tz="UTC"),
        )

    @staticmethod
    def standardise(df: pd.DataFrame, *, context: str = "DataSource") -> pd.DataFrame:
        """
        Validate ``df`` against the OHLCV contract and normalise its index to
        a tz-aware UTC ``DatetimeIndex``.

        Column checks are strict: all five OHLCV columns must be present, no
        others are allowed, all must be numeric, and the column index must be
        flat. Columns are returned in canonical order. The index must already be a
        ``DatetimeIndex`` without ``NaT`` or duplicate timestamps; naive
        timestamps are localised as UTC, aware timestamps are converted to
        UTC at nanosecond resolution (``datetime64[ns, UTC]``), and the result
        is sorted ascending with the index named ``Date``.

        Raises ``ValueError`` describing the first violation found.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError(
                f"{context}: provider must return a pandas DataFrame, "
                f"got {type(df).__name__}."
            )

        # ---- columns -----------------------------------------------------
        if isinstance(df.columns, pd.MultiIndex):
            raise ValueError(
                f"{context}: columns must be a flat Index of "
                f"{list(OHLCV_COLUMNS)}, got a MultiIndex {list(df.columns)}."
            )

        cols = [str(c) for c in df.columns]
        duplicates = sorted({c for c in cols if cols.count(c) > 1})
        if duplicates:
            raise ValueError(f"{context}: duplicate columns {duplicates}.")

        missing = [c for c in OHLCV_COLUMNS if c not in cols]
        if missing:
            raise ValueError(
                f"{context}: missing required columns {missing}; "
                f"expected exactly {list(OHLCV_COLUMNS)}, got {cols}."
            )

        extra = [c for c in cols if c not in OHLCV_COLUMNS]
        if extra:
            raise ValueError(
                f"{context}: unexpected columns {extra}; "
                f"expected exactly {list(OHLCV_COLUMNS)}."
            )

        out = df.loc[:, list(OHLCV_COLUMNS)]

        non_numeric = {c: str(out[c].dtype) for c in OHLCV_COLUMNS if not pd.api.types.is_numeric_dtype(out[c])}
        if non_numeric:
            raise ValueError(
                f"{context}: OHLCV columns must be numeric, got {non_numeric}."
            )

        # ---- index -------------------------------------------------------
        idx = out.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise ValueError(
                f"{context}: index must be a pandas DatetimeIndex, "
                f"got {type(idx).__name__}."
            )
        if idx.hasnans:
            raise ValueError(f"{context}: index contains NaT values.")
        if idx.has_duplicates:
            dupes = idx[idx.duplicated()].unique()[:5].tolist()
            raise ValueError(
                f"{context}: index contains duplicate timestamps, e.g. {dupes}."
            )

        try:
            idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"{context}: could not standardise index to UTC: {exc}"
            ) from exc
        # pandas >= 3 infers s/ms/us resolution from the input; pin to ns so the
        # dtype is ``datetime64[ns, UTC]`` whatever the provider or pandas version.
        idx = idx.as_unit("ns")

        out.index = idx
        if not out.index.is_monotonic_increasing:
            out = out.sort_index()
        # Identical frames regardless of origin (provider call, cache, slice):
        # no inferred-frequency metadata, canonical index name.
        out.index.freq = None
        out.index.name = "Date"
        return out

    # ------------------------------------------------------------- cleaning
    @staticmethod
    def infer_bar_size(index: pd.DatetimeIndex) -> Optional[pd.Timedelta]:
        """
        Bar size implied by ``index``: the most common positive spacing
        between consecutive timestamps (smallest one on a tie). ``None`` if it
        cannot be determined (fewer than two distinct timestamps).

        Using the mode rather than the minimum keeps a single spurious
        sub-bar timestamp from being mistaken for the bar size; using the mode
        rather than the provider's ``interval`` string keeps this independent
        of each provider's naming convention (``"D"`` vs ``"1d"`` etc).
        """
        if len(index) < 2:
            return None
        diffs = index.to_series().diff().dropna()
        diffs = diffs[diffs > pd.Timedelta(0)]
        if diffs.empty:
            return None
        return diffs.mode().min()

    def trading_days(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        """
        Expected timestamps of daily bars between ``start`` and ``end``
        (inclusive, both tz-aware UTC). One timestamp per Monday-Friday
        calendar day in ``calendar_tz``, at the same local wall-clock time as
        ``start`` -- so a provider stamping bars at local midnight stays on
        the grid across DST transitions.

        Override this with an exchange calendar to also exclude holidays;
        without one, holidays are indistinguishable from missing days and
        will be filled with a flat bar.
        """
        tz = self.calendar_tz
        first, last = start.tz_convert(tz), end.tz_convert(tz)
        days = pd.date_range(first.normalize().tz_localize(None), last.normalize().tz_localize(None), freq="D")
        days = days[days.day_of_week < 5]
        wall = days + (first - first.normalize())
        grid = wall.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")
        return grid[grid.notna()].tz_convert("UTC")

    def _intraday_grid(self, index: pd.DatetimeIndex, bar_size: pd.Timedelta) -> pd.DatetimeIndex:
        """
        Expected intraday timestamps. The observed index is split into sessions
        wherever the spacing exceeds ``max(session_gap, 2 * bar_size)``; each
        session is re-gridded at ``bar_size`` from its first to its last
        observed bar. Bars before a session's first trade cannot be inferred
        without a calendar and are not added.
        """
        threshold = max(self.session_gap, 2 * bar_size)
        series = index.to_series()
        session_id = (series.diff() > threshold).cumsum()
        pieces = [
            pd.date_range(session.iloc[0], session.iloc[-1], freq=bar_size)
            for _, session in series.groupby(session_id)
        ]
        return pieces[0].append(pieces[1:]) if pieces else pd.DatetimeIndex([], tz="UTC")

    def detect_gaps(
        self,
        df: pd.DataFrame,
        *,
        bar_size: Optional[pd.Timedelta] = None,
    ) -> pd.DatetimeIndex:
        """
        Timestamps at which a bar was expected but is absent from ``df``
        (which must already be standardised). Returns an empty index when the
        bar size cannot be inferred or is coarser than daily (weekly/monthly
        bars are not gap-filled: their spacing is irregular by nature).
        """
        idx = df.index
        empty = pd.DatetimeIndex([], tz="UTC")
        if len(idx) < 2:
            return empty
        bar_size = bar_size or self.infer_bar_size(idx)
        if bar_size is None or bar_size > _ONE_DAY:
            return empty
        if bar_size == _ONE_DAY:
            expected = self.trading_days(idx[0], idx[-1])
        else:
            expected = self._intraday_grid(idx, bar_size)
        return expected.as_unit("ns").difference(idx)

    def clean(self, df: pd.DataFrame, *, context: str = "DataSource") -> pd.DataFrame:
        """
        Fill missing bars in a standardised frame.

        For every timestamp reported by ``detect_gaps`` a synthetic flat bar is
        inserted: ``Close`` is the last observed close, ``Open/High/Low`` equal
        that close, ``Volume`` is 0. Observed rows are returned untouched and
        original dtypes are preserved. A summary is attached under
        ``df.attrs["gap_fill"]`` and logged.
        """
        if df.empty:
            return df

        bar_size = self.infer_bar_size(df.index)
        gaps = self.detect_gaps(df, bar_size=bar_size)
        if gaps.empty:
            return df

        full_index = df.index.union(gaps)
        out = df.reindex(full_index)

        # Forward-fill only into the synthetic rows; observed values stay as-is.
        out.loc[gaps, "Close"] = out["Close"].ffill().loc[gaps]
        for col in ("Open", "High", "Low"):
            out.loc[gaps, col] = out.loc[gaps, "Close"]
        out.loc[gaps, "Volume"] = 0

        # reindex() upcasts to float to hold NaN; restore the provider's dtypes.
        for col in OHLCV_COLUMNS:
            try:
                out[col] = out[col].astype(df[col].dtype)
            except (TypeError, ValueError):
                pass

        out.index.name = df.index.name
        ratio = len(gaps) / len(out)
        out.attrs = {
            **df.attrs,
            "gap_fill": {
                "bars_added": int(len(gaps)),
                "bar_size": str(bar_size),
                "fill_ratio": round(ratio, 4),
                "first_gap": gaps[0],
                "last_gap": gaps[-1],
            },
        }

        msg = (
            f"{context}: filled {len(gaps)} missing {bar_size} bar(s) "
            f"({ratio:.1%} of {len(out)}) with flat close / zero volume."
        )
        if ratio > 0.5:
            logger.warning("%s More synthetic than observed bars - check the "
                           "ticker's liquidity or the inferred bar size.", msg)
        else:
            logger.info(msg)
        return out