import math

import numpy as np
import pandas as pd
import pytest

import orchestrator
from brokers import FyersPaperBroker
from core.data.base import DataSource
from core.strategies.base import Strategy


# ---------------------------------------------------------------------------
# Test doubles (no network, deterministic)
# ---------------------------------------------------------------------------
class StaticDataSource(DataSource):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def fetch(self, symbol, start, end=None, interval="1d"):
        return self.df


class FixedSignalStrategy(Strategy):
    """Always emits the same latest signal; name has no regime restriction."""

    def __init__(self, signal: int, name: str = "Fixed"):
        super().__init__(name)
        self.signal = signal

    def generate_signals(self, df):
        out = df.copy()
        out["signal"] = self.signal
        return out


def make_broker(price: float, fees_pct: float = 0.0, cash: float = 100_000.0):
    return FyersPaperBroker(
        initial_cash=cash,
        market_price_provider=lambda symbol: price,
        fees_pct=fees_pct,
    )


def seed_healthy_history(strategy_name: str, n: int = 12):
    """Write enough profitable closed trades so the health gate passes."""
    now = pd.Timestamp.now()
    for i in range(n):
        orchestrator.append_trade_to_history(
            {
                "timestamp": now - pd.Timedelta(days=n - i),
                "strategy": strategy_name,
                "symbol": "SEED",
                "qty": 1,
                "side": "SELL",
                "price": 100.0,
                "entry_price": 95.0,
                "fees": 0.0,
                "pnl": 5.0 + i,
            }
        )


@pytest.fixture
def trade_log(tmp_path, monkeypatch):
    path = tmp_path / "paper_trade_log.csv"
    monkeypatch.setattr(orchestrator, "TRADE_LOG_CSV", str(path))
    return path


# ---------------------------------------------------------------------------
# Pure P&L helper
# ---------------------------------------------------------------------------
def test_compute_realised_pnl_gross_and_net():
    assert orchestrator.compute_realised_pnl(100.0, 110.0, 10) == 100.0
    assert orchestrator.compute_realised_pnl(100.0, 110.0, 10, fees=3.5) == 96.5
    assert orchestrator.compute_realised_pnl(100.0, 90.0, 10) == -100.0


# ---------------------------------------------------------------------------
# Full lifecycle through orchestrate_one_cycle
# ---------------------------------------------------------------------------
def test_entry_then_exit_records_numeric_pnl(sample_ohlcv, trade_log):
    strat_name = "Fixed"
    seed_healthy_history(strat_name)
    data = StaticDataSource(sample_ohlcv)

    # --- entry: long signal, no position ---
    broker = make_broker(price=100.0)
    orchestrator.orchestrate_one_cycle(FixedSignalStrategy(1, strat_name), "XYZ", broker, data)

    pos = broker.get_positions()["XYZ"]
    # sized as MAX_EXPOSURE_PER_TRADE of portfolio value (== cash here, nothing open yet)
    portfolio_value = orchestrator.compute_portfolio_exposure(broker)["portfolio_value"]
    expected_qty = math.floor(portfolio_value * orchestrator.MAX_EXPOSURE_PER_TRADE / sample_ohlcv["Close"].iloc[-1])
    assert pos["qty"] == expected_qty
    hist = orchestrator.load_trade_history()
    entry = hist[hist["symbol"] == "XYZ"].iloc[-1]
    assert entry["side"] == "BUY"
    assert entry["price"] == 100.0  # actual broker fill, not data-source close
    assert pd.isna(entry["pnl"])  # still open

    # --- exit: flat signal while holding -> realise P&L at new price ---
    broker._price_provider = lambda symbol: 110.0
    orchestrator.orchestrate_one_cycle(FixedSignalStrategy(0, strat_name), "XYZ", broker, data)

    assert "XYZ" not in broker.get_positions()
    hist = orchestrator.load_trade_history()
    exit_row = hist[hist["symbol"] == "XYZ"].iloc[-1]
    assert exit_row["side"] == "SELL"
    assert exit_row["qty"] == expected_qty
    assert exit_row["entry_price"] == 100.0
    assert exit_row["price"] == 110.0
    assert not pd.isna(exit_row["pnl"])
    assert exit_row["pnl"] == pytest.approx((110.0 - 100.0) * expected_qty)


def test_realised_pnl_is_net_of_entry_and_exit_fees(sample_ohlcv, trade_log):
    strat_name = "Fixed"
    seed_healthy_history(strat_name)
    data = StaticDataSource(sample_ohlcv)
    fees_pct = 0.05
    broker = make_broker(price=200.0, fees_pct=fees_pct)

    orchestrator.orchestrate_one_cycle(FixedSignalStrategy(1, strat_name), "XYZ", broker, data)
    qty = broker.get_positions()["XYZ"]["qty"]
    cash_before_exit = broker.get_balance()["cash"]

    broker._price_provider = lambda symbol: 190.0  # losing trade
    orchestrator.orchestrate_one_cycle(FixedSignalStrategy(0, strat_name), "XYZ", broker, data)

    exit_row = orchestrator.load_trade_history().iloc[-1]
    entry_fee = fees_pct / 100 * 200.0 * qty
    exit_fee = fees_pct / 100 * 190.0 * qty
    expected = (190.0 - 200.0) * qty - entry_fee - exit_fee
    assert exit_row["pnl"] == pytest.approx(expected)
    assert exit_row["fees"] == pytest.approx(entry_fee + exit_fee)
    # P&L must reconcile with the broker's actual cash movement over the round trip
    assert broker.get_balance()["cash"] - 100_000 == pytest.approx(expected)


def test_realised_pnl_feeds_strategy_metrics(sample_ohlcv, trade_log):
    strat_name = "Fixed"
    data = StaticDataSource(sample_ohlcv)
    # only 9 seeded trades -> below MIN_TRADES_FOR_STAT until our exit lands
    seed_healthy_history(strat_name, n=orchestrator.MIN_TRADES_FOR_STAT - 1)
    assert orchestrator.recent_metrics_from_history(strat_name)["sharpe"] is None

    # gate would block an entry, so open the position directly then exit via the orchestrator
    broker = make_broker(price=50.0)
    broker.place_order("XYZ", "BUY", 10)
    broker._price_provider = lambda symbol: 55.0
    orchestrator.orchestrate_one_cycle(FixedSignalStrategy(0, strat_name), "XYZ", broker, data)

    metrics = orchestrator.recent_metrics_from_history(strat_name)
    assert metrics["n_trades"] == orchestrator.MIN_TRADES_FOR_STAT
    assert metrics["sharpe"] is not None
    assert metrics["pf"] == float("inf")  # all winners, including our realised +50


def test_exit_bypasses_health_gate(sample_ohlcv, trade_log):
    """A position must always be closable even when the strategy has no stats."""
    data = StaticDataSource(sample_ohlcv)
    broker = make_broker(price=100.0)
    broker.place_order("XYZ", "BUY", 5)
    broker._price_provider = lambda symbol: 90.0

    orchestrator.orchestrate_one_cycle(FixedSignalStrategy(0, "NoHistory"), "XYZ", broker, data)

    assert "XYZ" not in broker.get_positions()
    row = orchestrator.load_trade_history().iloc[-1]
    assert row["side"] == "SELL"
    assert row["pnl"] == pytest.approx(-50.0)


def test_long_signal_while_in_position_does_not_pyramid(sample_ohlcv, trade_log):
    strat_name = "Fixed"
    seed_healthy_history(strat_name)
    data = StaticDataSource(sample_ohlcv)
    broker = make_broker(price=100.0)

    orchestrator.orchestrate_one_cycle(FixedSignalStrategy(1, strat_name), "XYZ", broker, data)
    qty_after_first = broker.get_positions()["XYZ"]["qty"]
    rows_after_first = len(orchestrator.load_trade_history())

    orchestrator.orchestrate_one_cycle(FixedSignalStrategy(1, strat_name), "XYZ", broker, data)

    assert broker.get_positions()["XYZ"]["qty"] == qty_after_first
    assert len(orchestrator.load_trade_history()) == rows_after_first


def test_flat_signal_without_position_does_nothing(sample_ohlcv, trade_log):
    data = StaticDataSource(sample_ohlcv)
    broker = make_broker(price=100.0)

    orchestrator.orchestrate_one_cycle(FixedSignalStrategy(0, "Fixed"), "XYZ", broker, data)

    assert broker.get_positions() == {}
    assert not trade_log.exists()


def test_rejected_order_is_not_logged(sample_ohlcv, trade_log):
    strat_name = "Fixed"
    seed_healthy_history(strat_name)
    data = StaticDataSource(sample_ohlcv)
    # price provider returns NaN -> broker rejects with "no-price"
    broker = make_broker(price=float("nan"))

    rows_before = len(orchestrator.load_trade_history())
    orchestrator.orchestrate_one_cycle(FixedSignalStrategy(1, strat_name), "XYZ", broker, data)

    assert broker.get_positions() == {}
    assert len(orchestrator.load_trade_history()) == rows_before


def test_legacy_log_without_new_columns_is_tolerated(trade_log):
    pd.DataFrame(
        [{"timestamp": pd.Timestamp.now(), "strategy": "Old", "symbol": "ABC",
          "side": "BUY", "qty": 1, "price": 10.0, "pnl": None}]
    ).to_csv(trade_log, index=False)

    hist = orchestrator.load_trade_history()
    assert {"entry_price", "fees"}.issubset(hist.columns)
    assert orchestrator.find_open_entry("Old", "ABC")["price"] == 10.0
    assert orchestrator.find_open_entry("Old", "ZZZ") is None


# ---------------------------------------------------------------------------
# Broker execution outcomes: only genuinely filled orders may be logged
# ---------------------------------------------------------------------------
class ScriptedBroker(FyersPaperBroker):
    """Paper broker whose place_order() returns a canned response (or raises)."""

    def __init__(self, response, **kw):
        super().__init__(market_price_provider=lambda s: 100.0, fees_pct=0.0, **kw)
        self.response = response
        self.calls = 0

    def place_order(self, symbol, side, qty, order_type="market", price=None):
        self.calls += 1
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.mark.parametrize(
    "response",
    [
        {"status": "rejected", "reason": "insufficient-cash"},
        {"status": "error", "error": "APIError(code=-2010)"},
        {"status": "REJECTED", "raw": {}},          # Binance exchange status
        {"status": "EXPIRED", "raw": {}},
        {"status": "CANCELED", "raw": {}},
        {"status": "NEW", "filled_qty": 0.0, "avg_fill_price": None},  # resting, unfilled
        {"status": "filled", "filled_qty": 0, "avg_fill_price": None},  # claims fill, no price
        {"status": "weird-unknown"},
        {},
        None,
        "not even a dict",
        RuntimeError("connection reset"),
    ],
    ids=lambda r: repr(r)[:40],
)
def test_unfilled_or_broken_entry_is_never_logged(sample_ohlcv, trade_log, response):
    strat_name = "Fixed"
    seed_healthy_history(strat_name)
    broker = ScriptedBroker(response)
    rows_before = len(orchestrator.load_trade_history())

    orchestrator.orchestrate_one_cycle(
        FixedSignalStrategy(1, strat_name), "XYZ", broker, StaticDataSource(sample_ohlcv)
    )

    assert broker.calls == 1
    assert len(orchestrator.load_trade_history()) == rows_before


@pytest.mark.parametrize(
    "response",
    [
        {"status": "rejected", "reason": "insufficient-qty"},
        {"status": "error", "error": "timeout"},
        {"status": "REJECTED"},
        None,
        RuntimeError("boom"),
    ],
    ids=lambda r: repr(r)[:40],
)
def test_unfilled_or_broken_exit_is_never_logged(sample_ohlcv, trade_log, response):
    broker = ScriptedBroker(response)
    broker.positions["XYZ"] = {"qty": 5.0, "avg_price": 100.0}  # open position

    orchestrator.orchestrate_one_cycle(
        FixedSignalStrategy(0, "Fixed"), "XYZ", broker, StaticDataSource(sample_ohlcv)
    )

    assert broker.calls == 1
    assert broker.get_positions()["XYZ"]["qty"] == 5.0  # still open
    assert not trade_log.exists()


def test_exchange_style_uppercase_filled_status_is_logged(sample_ohlcv, trade_log):
    """Binance passes through 'FILLED'; must be treated the same as 'filled'."""
    broker = ScriptedBroker({"status": "FILLED", "filled_qty": 5.0, "avg_fill_price": 120.0, "fee": 0.6})
    broker.positions["XYZ"] = {"qty": 5.0, "avg_price": 100.0}

    orchestrator.orchestrate_one_cycle(
        FixedSignalStrategy(0, "Fixed"), "XYZ", broker, StaticDataSource(sample_ohlcv)
    )

    row = orchestrator.load_trade_history().iloc[-1]
    assert row["side"] == "SELL"
    assert row["price"] == 120.0
    assert row["pnl"] == pytest.approx((120.0 - 100.0) * 5 - 0.6)


def test_partial_exit_logs_only_filled_portion_and_prorates_entry_fee(trade_log):
    strat = "Fixed"
    # entry of 10 @ 100 with a 10.0 entry fee
    orchestrator.append_trade_to_history({
        "timestamp": pd.Timestamp.now(), "strategy": strat, "symbol": "XYZ", "side": "BUY",
        "qty": 10.0, "price": 100.0, "entry_price": 100.0, "fees": 10.0, "pnl": np.nan,
    })
    position = {"qty": 10.0, "avg_price": 100.0}

    # first exit: 4 of 10 filled
    broker = ScriptedBroker({"status": "PARTIALLY_FILLED", "filled_qty": 4.0, "avg_fill_price": 110.0, "fee": 1.0})
    rec1 = orchestrator._exit_position(FixedSignalStrategy(0, strat), "XYZ", broker, position, 110.0)
    assert rec1["qty"] == 4.0
    assert rec1["fees"] == pytest.approx(10.0 * 4 / 10 + 1.0)
    assert rec1["pnl"] == pytest.approx((110 - 100) * 4 - 5.0)
    assert orchestrator.find_open_entry(strat, "XYZ") is not None  # still open

    # second exit: remaining 6 filled -> remaining 60% of entry fee realised
    broker = ScriptedBroker({"status": "filled", "filled_qty": 6.0, "avg_fill_price": 112.0, "fee": 1.5})
    rec2 = orchestrator._exit_position(FixedSignalStrategy(0, strat), "XYZ", broker, {"qty": 6.0, "avg_price": 100.0}, 112.0)
    assert rec2["fees"] == pytest.approx(10.0 * 6 / 10 + 1.5)
    assert rec2["pnl"] == pytest.approx((112 - 100) * 6 - 7.5)
    assert orchestrator.find_open_entry(strat, "XYZ") is None  # fully closed

    # across both legs the whole entry fee (and nothing more) was realised
    hist = orchestrator.load_trade_history()
    sells = hist[hist["side"] == "SELL"]
    assert sells["fees"].sum() == pytest.approx(10.0 + 1.0 + 1.5)


def test_interpret_order_result_classification():
    f = orchestrator.interpret_order_result
    assert f({"status": "filled", "filled_qty": 3, "avg_fill_price": 10, "fee": 0.1}, 3).status == "filled"
    # a 'filled' that delivered less than requested is a partial
    assert f({"status": "filled", "filled_qty": 2, "avg_fill_price": 10}, 3).status == "partially_filled"
    assert f({"status": "rejected", "reason": "x"}, 3).status == "rejected"
    assert f({"status": "error", "error": "x"}, 3).status == "error"
    assert f({"status": "FILLED", "filled_qty": 3, "avg_fill_price": 10}, 3).filled
    assert not f({"status": "NEW", "filled_qty": 0}, 3).filled
    assert not f(None, 3).filled
    assert not f({"status": "filled", "filled_qty": 3, "avg_fill_price": float("nan")}, 3).filled


# ---------------------------------------------------------------------------
# Portfolio-level exposure limit (MAX_TOTAL_EXPOSURE across all open positions)
# ---------------------------------------------------------------------------
class PricedBroker(FyersPaperBroker):
    """Paper broker with a per-symbol price table that tests can mutate."""

    def __init__(self, prices: dict, cash: float = 100_000.0):
        self.prices = dict(prices)
        super().__init__(initial_cash=cash, market_price_provider=lambda s: self.prices[s], fees_pct=0.0)


def run_entry(broker, data, symbol="XYZ", strat_name="Fixed"):
    orchestrator.orchestrate_one_cycle(FixedSignalStrategy(1, strat_name), symbol, broker, data)


def test_compute_portfolio_exposure_values_all_positions_at_market():
    broker = PricedBroker({"AAA": 100.0, "BBB": 50.0})
    broker.place_order("AAA", "BUY", 100)  # 10,000
    broker.place_order("BBB", "BUY", 100)  #  5,000 -> cash 85,000

    snap = orchestrator.compute_portfolio_exposure(broker)
    assert snap["cash"] == 85_000
    assert snap["positions_value"] == pytest.approx(15_000)
    assert snap["portfolio_value"] == pytest.approx(100_000)
    assert snap["exposure"] == pytest.approx(0.15)

    # price moves change exposure even with no new trades
    broker.prices["AAA"] = 200.0
    snap = orchestrator.compute_portfolio_exposure(broker)
    assert snap["positions_value"] == pytest.approx(25_000)
    assert snap["exposure"] == pytest.approx(25_000 / 110_000)


def test_compute_portfolio_exposure_with_no_positions_is_zero():
    broker = PricedBroker({})
    assert orchestrator.compute_portfolio_exposure(broker)["exposure"] == 0.0


def test_check_total_exposure_projection_and_boundary():
    broker = PricedBroker({"AAA": 100.0, "XYZ": 10.0})
    broker.place_order("AAA", "BUY", 200)  # 20% deployed

    ok, d = orchestrator.check_total_exposure(broker, "XYZ", qty=500, price=10.0)  # +5% -> 25% exactly
    assert ok and d["projected_exposure"] == pytest.approx(0.25)

    ok, d = orchestrator.check_total_exposure(broker, "XYZ", qty=501, price=10.0)  # 25.01%
    assert not ok and d["projected_exposure"] > 0.25


def test_entry_skipped_when_total_exposure_would_breach_limit(sample_ohlcv, trade_log):
    seed_healthy_history("Fixed")
    data = StaticDataSource(sample_ohlcv)
    broker = PricedBroker({"AAA": 100.0, "XYZ": float(sample_ohlcv["Close"].iloc[-1])})
    broker.place_order("AAA", "BUY", 220)  # 22% deployed; a 5%-of-portfolio trade would land at ~27%
    assert orchestrator.compute_portfolio_exposure(broker)["exposure"] == pytest.approx(0.22)

    rows_before = len(orchestrator.load_trade_history())
    run_entry(broker, data)

    assert "XYZ" not in broker.get_positions()           # order never placed
    assert broker.rejections == []                        # skipped by the gate, not the broker
    assert len(orchestrator.load_trade_history()) == rows_before
    assert orchestrator.compute_portfolio_exposure(broker)["exposure"] == pytest.approx(0.22)


def test_entry_allowed_when_total_exposure_stays_within_limit(sample_ohlcv, trade_log):
    seed_healthy_history("Fixed")
    data = StaticDataSource(sample_ohlcv)
    broker = PricedBroker({"AAA": 100.0, "XYZ": float(sample_ohlcv["Close"].iloc[-1])})
    broker.place_order("AAA", "BUY", 150)  # 15% deployed; +~4.25% of portfolio fits

    run_entry(broker, data)

    assert "XYZ" in broker.get_positions()
    assert orchestrator.compute_portfolio_exposure(broker)["exposure"] <= orchestrator.MAX_TOTAL_EXPOSURE


def test_exposure_gate_accumulates_across_many_symbols(sample_ohlcv, trade_log):
    """Each trade alone is only 5% of portfolio; together they must not exceed 25%."""
    seed_healthy_history("Fixed")
    data = StaticDataSource(sample_ohlcv)
    px = float(sample_ohlcv["Close"].iloc[-1])
    symbols = [f"S{i}" for i in range(10)]
    broker = PricedBroker({s: px for s in symbols})

    for s in symbols:
        run_entry(broker, data, symbol=s)

    exposure = orchestrator.compute_portfolio_exposure(broker)["exposure"]
    assert exposure <= orchestrator.MAX_TOTAL_EXPOSURE + 1e-9
    assert len(broker.get_positions()) < len(symbols)  # later signals were skipped
    assert len(broker.get_positions()) >= 5             # but not prematurely


def test_exposure_uses_market_value_not_cost_basis(sample_ohlcv, trade_log):
    """A position that has rallied consumes more of the limit than it cost."""
    seed_healthy_history("Fixed")
    data = StaticDataSource(sample_ohlcv)
    broker = PricedBroker({"AAA": 100.0, "XYZ": float(sample_ohlcv["Close"].iloc[-1])})
    broker.place_order("AAA", "BUY", 150)  # 15% at cost
    broker.prices["AAA"] = 200.0           # now 30,000 of a 115,000 portfolio ~ 26%

    run_entry(broker, data)

    assert "XYZ" not in broker.get_positions()


def test_exit_is_never_blocked_by_exposure_limit(sample_ohlcv, trade_log):
    data = StaticDataSource(sample_ohlcv)
    broker = PricedBroker({"XYZ": 100.0}, cash=10_000.0)
    broker.place_order("XYZ", "BUY", 90)  # 90% exposure, far above the limit
    assert orchestrator.compute_portfolio_exposure(broker)["exposure"] == pytest.approx(0.9)

    orchestrator.orchestrate_one_cycle(FixedSignalStrategy(0, "Fixed"), "XYZ", broker, data)

    assert "XYZ" not in broker.get_positions()
    assert orchestrator.load_trade_history().iloc[-1]["side"] == "SELL"


def test_exposure_falls_back_to_cost_basis_if_valuation_fails():
    broker = PricedBroker({"AAA": 100.0})
    broker.place_order("AAA", "BUY", 100)

    def boom(symbol):
        raise ConnectionError("quotes down")
    broker._price_provider = boom

    snap = orchestrator.compute_portfolio_exposure(broker)
    assert snap["positions_value"] == pytest.approx(10_000)  # 100 @ avg_price 100
    assert snap["exposure"] == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# Position sizing: % of total portfolio value, not % of spare cash
# ---------------------------------------------------------------------------
def test_compute_position_qty_uses_portfolio_value():
    # 5% of a 100k portfolio at 50/share -> 100 shares, whatever the cash split
    assert orchestrator.compute_position_qty(100_000, 50.0) == 100
    assert orchestrator.compute_position_qty(100_000, 50.0, cash_available=80_000) == 100
    assert orchestrator.compute_position_qty(100_000, 50.0, cash_available=20_000) == 100


def test_compute_position_qty_rounds_down_to_whole_shares():
    assert orchestrator.compute_position_qty(100_000, 333.0) == 15  # 5000 / 333 = 15.01


def test_compute_position_qty_is_capped_by_available_cash_including_fees():
    # 95% deployed: 5% of portfolio is 5000 but only 1000 cash remains.
    # 20 shares would be 1000 notional exactly -- with a 0.1% fee that costs 1001,
    # so the cap must back off to 19 (950 + 0.95 fee = 950.95 <= 1000).
    assert orchestrator.compute_position_qty(100_000, 50.0, cash_available=1_000) == 19
    assert orchestrator.compute_position_qty(100_000, 50.0, cash_available=0) == 0
    assert orchestrator.compute_position_qty(100_000, 50.0, cash_available=-5) == 0
    # fee-free broker: cap is on notional alone and the old answer is right
    assert orchestrator.compute_position_qty(100_000, 50.0, cash_available=1_000, fee_pct=0) == 20


def test_cash_cap_leaves_room_for_fee_at_exact_boundary():
    # cash 1001 covers 20 shares (1000) + 0.1% fee (1.0) exactly
    assert orchestrator.compute_position_qty(100_000, 50.0, cash_available=1_001) == 20
    assert orchestrator.compute_position_qty(100_000, 50.0, cash_available=1_000.99) == 19


def test_cash_cap_does_not_bind_when_cash_is_plentiful():
    # 5% of 100k = 5000 << 80k cash: fee allowance must not shave the size
    assert orchestrator.compute_position_qty(100_000, 50.0, cash_available=80_000) == 100


@pytest.mark.parametrize("cash", [1, 37, 99.99, 100, 1_000, 1_000.5, 12_345.67, 99_999])
@pytest.mark.parametrize("price", [0.5, 7.3, 50.0, 349.99, 2_500.0])
@pytest.mark.parametrize("fee_pct", [0.0, 0.05, 0.1])
def test_sized_order_always_settles_against_real_broker(cash, price, fee_pct):
    """Invariant: whatever the cash/price/fee, a sized order is never rejected for cash."""
    broker = FyersPaperBroker(initial_cash=cash, market_price_provider=lambda s: price, fees_pct=fee_pct)
    qty = orchestrator.compute_position_qty(
        cash, price, exposure_frac=1.0, cash_available=cash, fee_pct=orchestrator.estimated_fee_pct_for(broker)
    )
    if qty == 0:
        # genuinely unaffordable: one share + fee exceeds cash
        assert price * (1 + fee_pct / 100) > cash or price * (1 + orchestrator.ESTIMATED_FEE_PCT / 100) > cash
        return
    res = broker.place_order("X", "BUY", qty)
    assert res["status"] == "filled", res
    assert broker.get_balance()["cash"] >= -1e-9
    # and we did not leave a whole extra share on the table beyond the fee allowance
    assert (qty + 1) * price * (1 + orchestrator.ESTIMATED_FEE_PCT / 100) > cash


def test_estimated_fee_uses_broker_rate_when_it_is_higher():
    cheap = FyersPaperBroker(initial_cash=1, market_price_provider=lambda s: 1.0, fees_pct=0.05)
    pricey = FyersPaperBroker(initial_cash=1, market_price_provider=lambda s: 1.0, fees_pct=0.5)
    assert orchestrator.estimated_fee_pct_for(cheap) == orchestrator.ESTIMATED_FEE_PCT   # floor at 0.1%
    assert orchestrator.estimated_fee_pct_for(pricey) == 0.5                              # broker is dearer
    assert orchestrator.estimated_fee_pct_for(object()) == orchestrator.ESTIMATED_FEE_PCT  # unknown broker


def test_orchestrator_entry_at_exact_cash_boundary_is_filled_not_rejected(sample_ohlcv, trade_log, monkeypatch):
    """End-to-end: cash exactly equal to the notional of the per-trade slice used to
    be sized to the last rupee and bounced by the broker once fees were added."""
    seed_healthy_history("Fixed")
    px = float(sample_ohlcv["Close"].iloc[-1])
    monkeypatch.setattr(orchestrator, "MAX_TOTAL_EXPOSURE", 1.0)
    monkeypatch.setattr(orchestrator, "MAX_EXPOSURE_PER_TRADE", 1.0)
    cash = px * 10  # exactly ten shares' notional, zero slack for fees
    broker = FyersPaperBroker(initial_cash=cash, market_price_provider=lambda s: px, fees_pct=0.05)

    run_entry(broker, StaticDataSource(sample_ohlcv))

    assert broker.rejections == []
    assert broker.get_positions()["XYZ"]["qty"] == 9
    assert broker.get_balance()["cash"] >= 0


def test_compute_position_qty_custom_fraction():
    assert orchestrator.compute_position_qty(100_000, 50.0, exposure_frac=0.10) == 200


@pytest.mark.parametrize("pv,price", [(0, 50.0), (-1, 50.0), (100_000, 0.0), (100_000, -1.0), (None, 50.0), (100_000, None)])
def test_compute_position_qty_degenerate_inputs_give_zero(pv, price):
    assert orchestrator.compute_position_qty(pv, price) == 0


def test_orchestrator_sizes_from_portfolio_value_when_positions_are_open(sample_ohlcv, trade_log):
    """With 20% already deployed, the next trade must still be 5% of the *portfolio*,
    not 5% of the 80% of cash that remains."""
    seed_healthy_history("Fixed")
    data = StaticDataSource(sample_ohlcv)
    px = float(sample_ohlcv["Close"].iloc[-1])
    broker = PricedBroker({"AAA": 100.0, "XYZ": px})
    broker.place_order("AAA", "BUY", 200)  # 20,000 of 100,000
    assert broker.get_balance()["cash"] == 80_000

    run_entry(broker, data)

    qty = broker.get_positions()["XYZ"]["qty"]
    assert qty == math.floor(100_000 * orchestrator.MAX_EXPOSURE_PER_TRADE / px)   # portfolio-based
    assert qty != math.floor(80_000 * orchestrator.MAX_EXPOSURE_PER_TRADE / px)    # not cash-based
    assert orchestrator.compute_portfolio_exposure(broker)["exposure"] == pytest.approx(0.25, abs=0.005)


def test_successive_entries_are_equal_sized_as_cash_depletes(sample_ohlcv, trade_log):
    """Under cash-based sizing each trade shrank (5.0%, 4.6%, 4.5%...). Portfolio-based
    sizing keeps every slice the same."""
    seed_healthy_history("Fixed")
    data = StaticDataSource(sample_ohlcv)
    px = float(sample_ohlcv["Close"].iloc[-1])
    symbols = ["S1", "S2", "S3", "S4"]
    broker = PricedBroker({s: px for s in symbols})

    for s in symbols:
        run_entry(broker, data, symbol=s)

    qtys = [broker.get_positions()[s]["qty"] for s in symbols]
    assert len(set(qtys)) == 1, qtys
    assert qtys[0] == math.floor(100_000 * orchestrator.MAX_EXPOSURE_PER_TRADE / px)


def test_sizing_reflects_market_value_of_open_positions(sample_ohlcv, trade_log):
    """If holdings rally, the portfolio is bigger and the next 5% slice is bigger too."""
    seed_healthy_history("Fixed")
    data = StaticDataSource(sample_ohlcv)
    px = float(sample_ohlcv["Close"].iloc[-1])
    broker = PricedBroker({"AAA": 100.0, "XYZ": px})
    broker.place_order("AAA", "BUY", 100)  # 10,000
    broker.prices["AAA"] = 150.0           # now 15,000 -> portfolio 105,000

    run_entry(broker, data)

    assert broker.get_positions()["XYZ"]["qty"] == math.floor(105_000 * orchestrator.MAX_EXPOSURE_PER_TRADE / px)


def test_sizing_never_exceeds_available_cash(sample_ohlcv, trade_log, monkeypatch):
    """Exposure gate aside, a buy must never be sized beyond what cash can settle."""
    seed_healthy_history("Fixed")
    cheap = sample_ohlcv / 35.0                      # last Close ~ 9.97 so 100 cash buys ~10 shares
    px = float(cheap["Close"].iloc[-1])
    broker = PricedBroker({"AAA": 100.0, "XYZ": px}, cash=2_000.0)
    broker.place_order("AAA", "BUY", 19)             # 1,900 -> cash 100
    broker.prices["AAA"] = 5_000.0                   # positions 95,000; 5% of portfolio = 4,755 >> cash 100
    monkeypatch.setattr(orchestrator, "MAX_TOTAL_EXPOSURE", 1.0)  # isolate the cash cap from the gate

    run_entry(broker, StaticDataSource(cheap))

    assert broker.rejections == []                   # never asked the broker for an unaffordable order
    qty = broker.get_positions()["XYZ"]["qty"]
    assert qty == math.floor(100.0 / px)
    assert 0 < qty * px <= 100.0