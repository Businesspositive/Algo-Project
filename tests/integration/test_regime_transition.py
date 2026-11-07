"""
Integration tests: full `orchestrate_one_cycle` runs through regime changes,
forced flattening and settlement-aware risk gates, against the real paper
broker, the real strategies and the real CSV trade log (isolated per test).

The three behaviours the mechanism must guarantee are covered by the tests
marked REQUIREMENT 1 / 2 / 3:

  1. A regime change mid-trade issues a forced-flatten order that fills, and
     the resulting SELL / realised P&L are attributed to the strategy that
     opened the position -- not to the strategy routed for the new regime.
  2. If the broker rejects the forced-flatten order, the cycle halts: the
     position stays open and no capital is allocated to the new strategy.
  3. While a liquidation from the current cycle (or sweep) is unsettled, its
     in-flight value still counts toward the 25% total-exposure limit, so a
     same-cycle entry that would breach it is refused.
"""
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

import orchestrator
from core.strategies import StrategyRouter
from core.strategies.registry import get_strategy_class
from core.strategies.rsi_mean_reversion import RSIMeanReversion
from tests.support import (
    FixedRegimeDetector,
    ScriptedBroker,
    StaticDataSource,
    seed_healthy_history,
    symbol_rows,
    trend_frame,
)

PRICE = 230.0           # last-bar close of trend_frame("breakout")
LOT = 21                # 5% of 100,000 at 230 -> floor(5000/230)
LOT_VALUE = LOT * PRICE  # 4,830 = 4.83% of a 100k book


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def healthy_log(paper_trade_log):
    """Isolated trade log pre-seeded so every registry strategy passes the health gate."""
    for name in ("Donchian_Breakout", "Momentum", "RSI_MeanReversion"):
        seed_healthy_history(name)
    return paper_trade_log


@pytest.fixture
def router():
    return StrategyRouter()


@pytest.fixture
def broker():
    return ScriptedBroker(PRICE)


def run_cycle(broker, symbol, regime, router, *, frame="breakout", **kwargs):
    """One routed cycle for `symbol` under a forced regime label (None => detector raises)."""
    return orchestrator.orchestrate_one_cycle(
        None, symbol, broker, StaticDataSource(trend_frame(frame)),
        router=router, detector=FixedRegimeDetector(regime), **kwargs,
    )


def fill_book_to_limit(broker, router, fifth_leg_regime="NORMAL"):
    """Four Donchian legs + one fifth leg (Momentum by default): 5 x 4.83% = 24.15% deployed."""
    for sym in ("S1", "S2", "S3", "S4"):
        run_cycle(broker, sym, "TREND", router)
    run_cycle(broker, "MOM", fifth_leg_regime, router)
    assert len(broker.get_positions()) == 5
    assert orchestrator.compute_portfolio_exposure(broker)["exposure"] == pytest.approx(5 * LOT_VALUE / 100_000)


# ===========================================================================
# Routed entries
# ===========================================================================
def test_routed_entry_uses_regime_strategy_and_logs_owner(healthy_log, broker, router):
    run_cycle(broker, "XYZ", "TREND", router)
    assert broker.get_positions()["XYZ"]["qty"] == LOT
    rows = symbol_rows("XYZ")
    assert len(rows) == 1
    assert rows.iloc[0]["strategy"] == "Donchian_Breakout"
    assert rows.iloc[0]["side"] == "BUY"
    assert rows.iloc[0]["reason"] == orchestrator.REASON_SIGNAL
    assert orchestrator.find_position_owner("XYZ") == "Donchian_Breakout"


@pytest.mark.parametrize("regime", ["QUIET", "NEVER_HEARD_OF_IT", None])
def test_cash_regime_while_flat_places_no_orders(healthy_log, broker, router, regime):
    run_cycle(broker, "XYZ", regime, router)
    assert broker.orders == []
    assert broker.get_positions() == {}
    assert symbol_rows("XYZ").empty


def test_range_regime_routes_to_rsi_not_donchian(healthy_log, broker, router, monkeypatch):
    seen = []
    real = RSIMeanReversion.generate_signals
    monkeypatch.setattr(RSIMeanReversion, "generate_signals", lambda self, df: seen.append(self.name) or real(self, df))
    run_cycle(broker, "XYZ", "RANGE", router)
    assert seen == ["RSI_MeanReversion"]
    assert broker.orders == []  # RSI is overbought on a rising series -> no entry


# ===========================================================================
# REQUIREMENT 1 -- regime change mid-trade: forced flatten, P&L to the opener
# ===========================================================================
def test_REQ1_hard_regime_shift_mid_trade_force_flattens_and_attributes_pnl_to_opener(
    healthy_log, broker, router, monkeypatch
):
    # Open via Donchian in TREND at 230
    run_cycle(broker, "XYZ", "TREND", router)
    assert broker.get_positions()["XYZ"]["qty"] == LOT

    # Regime flips to RANGE while the price has moved: RSI_MeanReversion is
    # now the routed strategy, Donchian is not eligible in RANGE.
    broker.prices["XYZ"] = 250.0
    rsi_calls = []
    real = RSIMeanReversion.generate_signals
    monkeypatch.setattr(RSIMeanReversion, "generate_signals", lambda self, df: rsi_calls.append(1) or real(self, df))

    run_cycle(broker, "XYZ", "RANGE", router)

    # (a) a market SELL for the full position was issued and filled
    assert broker.sells() == [("XYZ", "SELL", float(LOT))]
    assert "XYZ" not in broker.get_positions()

    # (b) the SELL is logged under the ORIGINAL strategy with the regime-shift reason
    rows = symbol_rows("XYZ")
    assert list(rows["side"]) == ["BUY", "SELL"]
    assert list(rows["strategy"]) == ["Donchian_Breakout", "Donchian_Breakout"]
    assert rows.iloc[1]["reason"] == f"{orchestrator.REASON_REGIME_SHIFT}:RANGE"

    # (c) realised P&L is numeric, correct, and paired with the original entry
    assert rows.iloc[1]["entry_price"] == PRICE
    assert rows.iloc[1]["pnl"] == pytest.approx((250.0 - PRICE) * LOT)
    assert orchestrator.find_position_owner("XYZ") is None  # round trip fully closed

    # (d) P&L feeds Donchian's metrics and NOT RSI's
    donchian = orchestrator.load_trade_history().query("strategy == 'Donchian_Breakout' and symbol == 'XYZ'")
    assert donchian["pnl"].dropna().sum() == pytest.approx((250.0 - PRICE) * LOT)
    rsi = orchestrator.load_trade_history().query("strategy == 'RSI_MeanReversion' and symbol == 'XYZ'")
    assert rsi.empty

    # (e) the newly routed strategy was only consulted AFTER the flatten
    assert rsi_calls == [1]


def test_REQ1_same_cycle_reallocation_orders_flatten_before_new_entry(healthy_log, broker, router):
    """Momentum long in NORMAL; regime -> VOL_BREAKOUT where only Donchian trades:
    old leg is sold first, then Donchian enters, in that order, within one cycle."""
    run_cycle(broker, "XYZ", "NORMAL", router)
    assert orchestrator.find_position_owner("XYZ") == "Momentum"

    run_cycle(broker, "XYZ", "VOL_BREAKOUT", router)

    assert broker.orders[-2:] == [("XYZ", "SELL", float(LOT)), ("XYZ", "BUY", float(LOT))]
    rows = symbol_rows("XYZ")
    assert list(zip(rows["strategy"], rows["side"], rows["reason"])) == [
        ("Momentum", "BUY", orchestrator.REASON_SIGNAL),
        ("Momentum", "SELL", f"{orchestrator.REASON_REGIME_SHIFT}:VOL_BREAKOUT"),
        ("Donchian_Breakout", "BUY", orchestrator.REASON_SIGNAL),
    ]
    assert orchestrator.find_position_owner("XYZ") == "Donchian_Breakout"


@pytest.mark.parametrize("regime, label", [("QUIET", "QUIET"), (None, "None"), ("UNKNOWN_LABEL", "UNKNOWN_LABEL")])
def test_REQ1_cash_regime_mid_trade_force_flattens_and_runs_no_strategy(healthy_log, broker, router, regime, label):
    run_cycle(broker, "XYZ", "TREND", router)
    run_cycle(broker, "XYZ", regime, router)

    assert broker.sells() == [("XYZ", "SELL", float(LOT))]
    assert len(broker.buys()) == 1                         # no re-entry
    assert "XYZ" not in broker.get_positions()
    rows = symbol_rows("XYZ")
    assert rows.iloc[-1]["strategy"] == "Donchian_Breakout"
    assert rows.iloc[-1]["reason"] == f"{orchestrator.REASON_REGIME_CASH}:{label}"
    assert not np.isnan(rows.iloc[-1]["pnl"])


def test_soft_shift_owner_still_eligible_is_not_flattened(healthy_log, broker, router):
    """Momentum opened in NORMAL; TREND also lists Momentum (2nd priority) ->
    the owner keeps the position even though Donchian is TREND's first choice."""
    run_cycle(broker, "XYZ", "NORMAL", router)
    orders_before = list(broker.orders)
    run_cycle(broker, "XYZ", "TREND", router)
    assert broker.orders == orders_before
    assert orchestrator.find_position_owner("XYZ") == "Momentum"
    assert broker.get_positions()["XYZ"]["qty"] == LOT


def test_owner_exits_on_its_own_signal_with_signal_reason(healthy_log, broker, router):
    run_cycle(broker, "XYZ", "TREND", router)
    broker.prices["XYZ"] = 150.0
    run_cycle(broker, "XYZ", "TREND", router, frame="collapse")   # Donchian signal 0
    rows = symbol_rows("XYZ")
    assert rows.iloc[-1]["side"] == "SELL"
    assert rows.iloc[-1]["reason"] == orchestrator.REASON_SIGNAL
    assert rows.iloc[-1]["pnl"] == pytest.approx((150.0 - PRICE) * LOT)


def test_unattributed_position_is_flattened_under_stand_in_then_routed_strategy_may_enter(
    healthy_log, broker, router
):
    broker.place_order("XYZ", "BUY", 10)   # opened out-of-band: no BUY in the trade log
    run_cycle(broker, "XYZ", "TREND", router)

    assert broker.orders[1:] == [("XYZ", "SELL", 10.0), ("XYZ", "BUY", float(LOT))]
    rows = symbol_rows("XYZ")
    assert list(zip(rows["strategy"], rows["side"], rows["reason"])) == [
        (orchestrator.UNATTRIBUTED_STRATEGY, "SELL", orchestrator.REASON_UNATTRIBUTED),
        ("Donchian_Breakout", "BUY", orchestrator.REASON_SIGNAL),
    ]
    # nothing leaked into a real strategy's P&L
    assert orchestrator.load_trade_history().query("strategy == 'Donchian_Breakout'")["pnl"].dropna().sum() == \
        pytest.approx(sum(5.0 + i for i in range(12)))  # only the seeded history


def test_force_flatten_false_lets_ineligible_owner_keep_managing(healthy_log, broker, router):
    run_cycle(broker, "XYZ", "TREND", router)
    orders_before = list(broker.orders)
    run_cycle(broker, "XYZ", "RANGE", router, force_flatten=False)
    assert broker.orders == orders_before
    assert orchestrator.find_position_owner("XYZ") == "Donchian_Breakout"


def test_pinned_strategy_ineligible_in_regime_is_flattened_without_reentry(healthy_log, broker):
    pinned = get_strategy_class("Donchian_Breakout")()
    data = StaticDataSource(trend_frame("breakout"))
    orchestrator.orchestrate_one_cycle(pinned, "XYZ", broker, data, detector=FixedRegimeDetector("TREND"))
    assert broker.get_positions()["XYZ"]["qty"] == LOT

    orchestrator.orchestrate_one_cycle(pinned, "XYZ", broker, data, detector=FixedRegimeDetector("RANGE"))
    assert broker.sells() == [("XYZ", "SELL", float(LOT))]
    assert len(broker.buys()) == 1
    assert "XYZ" not in broker.get_positions()


def test_flatten_all_positions_kill_switch(healthy_log, broker, router):
    run_cycle(broker, "XYZ", "TREND", router)
    run_cycle(broker, "ABC", "NORMAL", router)
    broker.place_order("OOB", "BUY", 3)
    ledger = orchestrator.SettlementLedger()

    results = orchestrator.flatten_all_positions(broker, router, ledger=ledger)

    assert set(results) == {"XYZ", "ABC", "OOB"} and all(r.closed for r in results.values())
    assert broker.get_positions() == {}
    assert len(ledger) == 3
    tail = orchestrator.load_trade_history().tail(3)
    assert set(tail["reason"]) == {orchestrator.REASON_MANUAL}
    assert set(tail["strategy"]) == {"Donchian_Breakout", "Momentum", orchestrator.UNATTRIBUTED_STRATEGY}


# ===========================================================================
# REQUIREMENT 2 -- rejected / partial forced flatten halts the cycle
# ===========================================================================
def test_REQ2_rejected_flatten_halts_cycle_and_allocates_nothing(healthy_log, broker, router, monkeypatch):
    run_cycle(broker, "XYZ", "NORMAL", router)             # Momentum long
    rows_before = len(orchestrator.load_trade_history())
    cash_before = broker.get_balance()["cash"]

    # The new regime's strategy must never even be consulted.
    donchian_cls = get_strategy_class("Donchian_Breakout")
    monkeypatch.setattr(donchian_cls, "generate_signals",
                        lambda self, df: pytest.fail("new strategy evaluated despite halted cycle"))

    broker.reject_sells = True
    ledger = orchestrator.SettlementLedger()
    run_cycle(broker, "XYZ", "VOL_BREAKOUT", router, ledger=ledger)

    # the flatten was attempted (and only it)...
    assert broker.orders[-1] == ("XYZ", "SELL", float(LOT))
    assert len(broker.buys()) == 1
    # ...but nothing changed: position intact, log untouched, no cash moved, nothing ledgered
    assert broker.get_positions()["XYZ"]["qty"] == LOT
    assert orchestrator.find_position_owner("XYZ") == "Momentum"
    assert len(orchestrator.load_trade_history()) == rows_before
    assert broker.get_balance()["cash"] == cash_before
    assert len(ledger) == 0


def test_REQ2_rejected_flatten_is_retried_next_cycle_and_then_reallocates(healthy_log, broker, router):
    run_cycle(broker, "XYZ", "NORMAL", router)
    broker.reject_sells = True
    run_cycle(broker, "XYZ", "VOL_BREAKOUT", router)
    assert broker.get_positions()["XYZ"]["qty"] == LOT

    broker.reject_sells = False
    run_cycle(broker, "XYZ", "VOL_BREAKOUT", router)       # fresh ledger: liquidation settled
    assert broker.orders[-2:] == [("XYZ", "SELL", float(LOT)), ("XYZ", "BUY", float(LOT))]
    assert orchestrator.find_position_owner("XYZ") == "Donchian_Breakout"


def test_REQ2_partial_flatten_halts_cycle_keeps_owner_and_ledgers_sold_slice(healthy_log, broker, router):
    run_cycle(broker, "XYZ", "NORMAL", router)
    donchian_cls = get_strategy_class("Donchian_Breakout")

    broker.partial_sell_qty = 10
    ledger = orchestrator.SettlementLedger()
    with pytest.MonkeyPatch.context() as mp:   # scoped: must not undo the trade-log redirect
        mp.setattr(donchian_cls, "generate_signals",
                   lambda self, df: pytest.fail("new strategy evaluated despite halted cycle"))
        run_cycle(broker, "XYZ", "VOL_BREAKOUT", router, ledger=ledger)

    assert broker.get_positions()["XYZ"]["qty"] == LOT - 10
    assert len(broker.buys()) == 1
    rows = symbol_rows("XYZ")
    assert rows.iloc[-1]["side"] == "SELL" and rows.iloc[-1]["qty"] == 10
    assert rows.iloc[-1]["strategy"] == "Momentum"
    assert orchestrator.find_position_owner("XYZ") == "Momentum"   # remainder still owned
    assert len(ledger) == 1 and ledger.pending()[0].qty == 10      # sold slice is unsettled

    # next cycle: the remainder is closed under the same owner, then the new strategy enters
    broker.partial_sell_qty = None
    run_cycle(broker, "XYZ", "VOL_BREAKOUT", router)
    rows = symbol_rows("XYZ")
    assert list(zip(rows["strategy"], rows["side"], rows["qty"])) == [
        ("Momentum", "BUY", float(LOT)),
        ("Momentum", "SELL", 10.0),
        ("Momentum", "SELL", float(LOT - 10)),
        ("Donchian_Breakout", "BUY", float(LOT)),
    ]


def test_REQ2_broker_exception_on_flatten_halts_cycle(healthy_log, broker, router):
    run_cycle(broker, "XYZ", "NORMAL", router)

    def explode(symbol, side, qty, **kw):
        raise ConnectionError("socket closed")

    broker.place_order = explode
    run_cycle(broker, "XYZ", "VOL_BREAKOUT", router)   # must not raise
    assert broker.get_positions()["XYZ"]["qty"] == LOT
    assert orchestrator.find_position_owner("XYZ") == "Momentum"


# ===========================================================================
# REQUIREMENT 3 -- 25% limit respected with in-flight liquidation value
# ===========================================================================
def test_REQ3_same_cycle_entry_blocked_when_in_flight_value_would_breach_25pct(healthy_log, broker, router, capsys):
    fill_book_to_limit(broker, router)                      # 24.15% deployed, fifth leg = Momentum
    capsys.readouterr()

    # Hard shift on the Momentum leg: it is sold, Donchian wants in at 4.83%.
    # Positions (19.32%) + in-flight (4.83%) + new (4.83%) = 28.98% > 25%.
    run_cycle(broker, "MOM", "VOL_BREAKOUT", router)

    assert broker.orders[-1] == ("MOM", "SELL", float(LOT))  # flatten happened, no BUY followed
    assert "MOM" not in broker.get_positions()
    assert len(broker.get_positions()) == 4
    out = capsys.readouterr().out
    assert "in-flight 4830.00" in out
    assert "projected with this trade 29.0%" in out
    assert "Total exposure gate failed" in out

    # Once the liquidation is considered settled (fresh ledger next cycle) the entry is admitted.
    run_cycle(broker, "MOM", "VOL_BREAKOUT", router)
    assert orchestrator.find_position_owner("MOM") == "Donchian_Breakout"
    assert orchestrator.compute_portfolio_exposure(broker)["exposure"] <= 0.25 + 1e-9


def test_REQ3_without_settlement_awareness_the_same_entry_would_have_passed(healthy_log, broker, router):
    """Control: proves the block above comes from the ledger, not from ordinary exposure maths."""
    fill_book_to_limit(broker, router)
    broker.place_order("MOM", "SELL", LOT)                  # liquidate outside the orchestrator
    ok, d = orchestrator.check_total_exposure(broker, "MOM", LOT, PRICE)
    assert ok and d["projected_exposure"] == pytest.approx(5 * LOT_VALUE / 100_000)
    liq = orchestrator.PendingLiquidation("MOM", LOT, PRICE, LOT_VALUE, pre_qty=LOT)
    ok, d = orchestrator.check_total_exposure(broker, "MOM", LOT, PRICE, pending=liq)
    assert not ok and d["projected_exposure"] == pytest.approx(6 * LOT_VALUE / 100_000)


def test_REQ3_shared_ledger_blocks_other_symbols_in_same_sweep_until_settled(healthy_log, broker, router):
    fill_book_to_limit(broker, router)
    sweep = orchestrator.SettlementLedger()

    run_cycle(broker, "MOM", "QUIET", router, ledger=sweep)   # regime_cash flatten, recorded
    assert len(sweep) == 1 and sweep.pending()[0].symbol == "MOM"
    assert sweep.unsettled_proceeds == pytest.approx(LOT_VALUE)

    run_cycle(broker, "ABC", "TREND", router, ledger=sweep)   # wants 4.83% -> 28.98% with in-flight
    assert "ABC" not in broker.get_positions()
    assert len(broker.buys()) == 5

    sweep.settle()
    run_cycle(broker, "ABC", "TREND", router, ledger=sweep)
    assert broker.get_positions()["ABC"]["qty"] == LOT


def test_REQ3_t_plus_one_ledger_releases_budget_only_after_delay(healthy_log, broker, router):
    fill_book_to_limit(broker, router)
    ledger = orchestrator.SettlementLedger(settlement_delay=timedelta(days=1))
    run_cycle(broker, "MOM", "QUIET", router, ledger=ledger)

    run_cycle(broker, "ABC", "TREND", router, ledger=ledger)
    assert "ABC" not in broker.get_positions()

    ledger.pending()[0].timestamp -= pd.Timedelta(days=1, minutes=1)  # time passes
    run_cycle(broker, "ABC", "TREND", router, ledger=ledger)
    assert "ABC" in broker.get_positions()


def test_REQ3_unsettled_proceeds_are_withheld_from_position_sizing(healthy_log, router, monkeypatch):
    """With cash as the binding constraint, the transition-cycle order is sized
    from settled cash only -- the freed proceeds cannot fund it."""
    broker = ScriptedBroker(PRICE, cash=LOT_VALUE + 1_170.0)   # one Momentum lot + 1,170 spare cash
    broker.place_order("MOM", "BUY", LOT)
    orchestrator.append_trade_to_history(
        {"timestamp": pd.Timestamp.now(), "strategy": "Momentum", "symbol": "MOM", "side": "BUY",
         "qty": LOT, "price": PRICE, "entry_price": PRICE, "fees": 0.0, "pnl": float("nan"),
         "reason": orchestrator.REASON_SIGNAL}
    )
    assert broker.get_balance()["cash"] == pytest.approx(1_170.0)

    # make cash -- not the 5% / 25% limits -- the binding constraint for the transition entry
    monkeypatch.setattr(orchestrator, "MAX_EXPOSURE_PER_TRADE", 0.90)
    monkeypatch.setattr(orchestrator, "MAX_TOTAL_EXPOSURE", 1.0)
    run_cycle(broker, "MOM", "VOL_BREAKOUT", router)           # flatten frees 4,830, but it is unsettled

    assert broker.orders[-1][1] == "BUY"
    assert broker.get_positions()["MOM"]["qty"] == 5            # floor(1170 / 230), not floor(6000 / 230) == 26


def test_REQ3_entry_at_limit_passes_once_nothing_is_in_flight(healthy_log, broker, router):
    """Sanity: the in-flight logic never blocks a plain entry that fits."""
    for sym in ("S1", "S2", "S3", "S4"):
        run_cycle(broker, sym, "TREND", router)
    run_cycle(broker, "S5", "TREND", router)
    assert len(broker.get_positions()) == 5
    run_cycle(broker, "S6", "TREND", router)                    # 6th would be 28.98% -> blocked normally
    assert "S6" not in broker.get_positions()


# ===========================================================================
# Trade-log schema
# ===========================================================================
def test_legacy_trade_log_without_reason_column_is_tolerated(paper_trade_log):
    pd.DataFrame(
        [{"timestamp": pd.Timestamp("2025-01-01"), "strategy": "Momentum", "symbol": "XYZ",
          "side": "BUY", "qty": 5, "price": 100.0}]
    ).to_csv(paper_trade_log, index=False)
    hist = orchestrator.load_trade_history()
    assert "reason" in hist.columns and hist["reason"].isna().all()
    assert orchestrator.find_position_owner("XYZ") == "Momentum"