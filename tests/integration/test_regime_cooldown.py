"""
Integration tests for the regime-transition cooldown inside
`orchestrate_one_cycle`: a regime flickering on a detector threshold must not
churn sell/re-buy fees every day, while capital protection (immediate forced
flatten of an ineligible owner) is untouched.
"""
import pandas as pd
import pytest

import orchestrator
from core.strategies import StrategyRouter
from core.strategies.registry import get_strategy_class
from tests.support import (
    FixedRegimeDetector,
    ScriptedBroker,
    StaticDataSource,
    seed_healthy_history,
    symbol_rows,
    trend_frame,
)

PRICE = 230.0
LOT = 21


@pytest.fixture
def healthy_log(paper_trade_log):
    for name in ("Donchian_Breakout", "Momentum", "RSI_MeanReversion"):
        seed_healthy_history(name)
    return paper_trade_log


@pytest.fixture
def router():
    return StrategyRouter()


@pytest.fixture
def broker():
    return ScriptedBroker(PRICE, fees_pct=0.1)   # fees on, so churn has a measurable cost


def run_cycle(broker, symbol, history, router, **kwargs):
    """One routed cycle where the detector's per-bar history ends with `history`."""
    return orchestrator.orchestrate_one_cycle(
        None, symbol, broker, StaticDataSource(trend_frame("breakout")),
        router=router, detector=FixedRegimeDetector(history=history), **kwargs,
    )


def fees_paid(broker):
    return sum(t["fee"] for t in broker.trade_log)


# ---------------------------------------------------------------------------
# The churn problem, and that the cooldown fixes it
# ---------------------------------------------------------------------------
FLICKER = ["RANGE", "TREND", "RANGE", "TREND", "RANGE", "TREND"]
# Donchian trades TREND, RSI_MeanReversion trades RANGE -- mutually exclusive,
# so a Donchian holder sees a hard shift on every RANGE day and (without a
# cooldown) re-enters on every TREND day.  RSI never fires on this data.


def run_flicker(broker, router, **kwargs):
    run_cycle(broker, "XYZ", ["TREND"] * 3, router, **kwargs)        # Donchian enters in a stable TREND
    assert orchestrator.find_position_owner("XYZ") == "Donchian_Breakout"
    window = ["TREND"] * 3
    for label in FLICKER:
        window = (window + [label])[-3:]
        run_cycle(broker, "XYZ", window, router, **kwargs)


def test_without_cooldown_a_flickering_regime_churns_every_day(healthy_log, broker, router):
    run_flicker(broker, router, min_stable_periods=1)
    # every RANGE day: flatten Donchian; every TREND day: Donchian re-enters
    assert len(broker.sells()) == 3
    assert len(broker.buys()) == 4
    assert fees_paid(broker) == pytest.approx(7 * LOT * PRICE * 0.001)
    reasons = list(symbol_rows("XYZ")["reason"])
    assert reasons.count(f"{orchestrator.REASON_REGIME_SHIFT}:RANGE") == 3


def test_with_cooldown_the_same_flicker_costs_one_protective_exit(healthy_log, broker, router):
    run_flicker(broker, router)                                        # default: 3 stable bars
    # day 1 (RANGE x1): Donchian is ineligible -> flattened at once.  After that the
    # regime never holds 3 bars, so nobody is allowed to open a new position.
    assert len(broker.sells()) == 1
    assert len(broker.buys()) == 1
    assert broker.get_positions() == {}
    assert fees_paid(broker) == pytest.approx(2 * LOT * PRICE * 0.001)
    rows = symbol_rows("XYZ")
    assert list(zip(rows["strategy"], rows["side"], rows["reason"])) == [
        ("Donchian_Breakout", "BUY", orchestrator.REASON_SIGNAL),
        ("Donchian_Breakout", "SELL", f"{orchestrator.REASON_REGIME_SHIFT}:RANGE"),
    ]


def test_cooldown_reduces_fees_versus_no_cooldown(healthy_log, router):
    churned, calm = ScriptedBroker(PRICE, fees_pct=0.1), ScriptedBroker(PRICE, fees_pct=0.1)
    run_flicker(churned, router, min_stable_periods=1)
    healthy_log.unlink()
    for name in ("Donchian_Breakout", "Momentum", "RSI_MeanReversion"):
        seed_healthy_history(name)
    run_flicker(calm, router)
    assert fees_paid(calm) < fees_paid(churned)
    assert len(calm.orders) < len(churned.orders)


# ---------------------------------------------------------------------------
# Day-by-day behaviour of a transition
# ---------------------------------------------------------------------------
def test_new_regime_must_hold_three_bars_before_new_capital_is_allocated(healthy_log, broker, router, capsys):
    run_cycle(broker, "XYZ", ["NORMAL"] * 3, router)                  # Momentum long
    capsys.readouterr()

    # bar 1 of VOL_BREAKOUT: protective flatten now, Donchian (signal=1) NOT allowed in
    run_cycle(broker, "XYZ", ["NORMAL", "NORMAL", "VOL_BREAKOUT"], router)
    assert broker.orders[-1] == ("XYZ", "SELL", float(LOT))
    assert broker.get_positions() == {}
    out = capsys.readouterr().out
    assert "VOL_BREAKOUT [DIRECTIONAL] x1/3 (cooldown, 2 more bar(s) needed, was NORMAL)" in out
    assert "no new positions" in out

    # bar 2: still waiting
    run_cycle(broker, "XYZ", ["NORMAL", "VOL_BREAKOUT", "VOL_BREAKOUT"], router)
    assert broker.get_positions() == {}
    assert "x2/3 (cooldown, 1 more bar(s) needed" in capsys.readouterr().out

    # bar 3: regime confirmed -> Donchian enters
    run_cycle(broker, "XYZ", ["VOL_BREAKOUT"] * 3, router)
    assert broker.orders[-1] == ("XYZ", "BUY", float(LOT))
    assert orchestrator.find_position_owner("XYZ") == "Donchian_Breakout"
    assert "x3/3 (stable)" in capsys.readouterr().out


def test_flip_back_resets_the_clock(healthy_log, broker, router):
    run_cycle(broker, "XYZ", ["TREND", "TREND", "RANGE"], router)
    run_cycle(broker, "XYZ", ["TREND", "RANGE", "RANGE"], router)
    run_cycle(broker, "XYZ", ["RANGE", "RANGE", "TREND"], router)   # back to TREND: x1
    run_cycle(broker, "XYZ", ["RANGE", "TREND", "TREND"], router)   # x2
    assert broker.orders == []
    run_cycle(broker, "XYZ", ["TREND", "TREND", "TREND"], router)   # x3 -> go
    assert broker.buys() == [("XYZ", "BUY", float(LOT))]


def test_custom_min_stable_periods(healthy_log, broker, router):
    run_cycle(broker, "XYZ", ["RANGE", "TREND", "TREND", "TREND", "TREND"], router, min_stable_periods=5)
    assert broker.orders == []
    run_cycle(broker, "XYZ", ["TREND"] * 5, router, min_stable_periods=5)
    assert len(broker.buys()) == 1


def test_min_stable_periods_one_restores_same_cycle_reallocation(healthy_log, broker, router):
    run_cycle(broker, "XYZ", ["NORMAL"] * 3, router)
    run_cycle(broker, "XYZ", ["NORMAL", "NORMAL", "VOL_BREAKOUT"], router, min_stable_periods=1)
    assert broker.orders[-2:] == [("XYZ", "SELL", float(LOT)), ("XYZ", "BUY", float(LOT))]
    assert orchestrator.find_position_owner("XYZ") == "Donchian_Breakout"


# ---------------------------------------------------------------------------
# Capital protection is never delayed by the cooldown
# ---------------------------------------------------------------------------
def test_ineligible_owner_is_flattened_on_the_first_bar_of_a_new_regime(healthy_log, broker, router):
    run_cycle(broker, "XYZ", ["TREND"] * 3, router)                   # Donchian long
    broker.prices["XYZ"] = 200.0
    run_cycle(broker, "XYZ", ["TREND", "TREND", "RANGE"], router)     # RANGE x1: Donchian ineligible
    rows = symbol_rows("XYZ")
    assert rows.iloc[-1]["side"] == "SELL"
    assert rows.iloc[-1]["strategy"] == "Donchian_Breakout"
    assert rows.iloc[-1]["reason"] == f"{orchestrator.REASON_REGIME_SHIFT}:RANGE"
    assert rows.iloc[-1]["pnl"] == pytest.approx((200.0 - PRICE) * LOT - rows.iloc[-1]["fees"])
    assert broker.get_positions() == {}


def test_cash_regime_flicker_still_flattens_immediately(healthy_log, broker, router):
    run_cycle(broker, "XYZ", ["TREND"] * 3, router)
    run_cycle(broker, "XYZ", ["TREND", "TREND", "QUIET"], router)
    assert broker.get_positions() == {}
    assert symbol_rows("XYZ").iloc[-1]["reason"] == f"{orchestrator.REASON_REGIME_CASH}:QUIET"


def test_soft_shift_during_cooldown_holds_without_orders(healthy_log, broker, router):
    run_cycle(broker, "XYZ", ["NORMAL"] * 3, router)                  # Momentum long
    orders_before = list(broker.orders)
    for window in (["NORMAL", "NORMAL", "TREND"], ["NORMAL", "TREND", "NORMAL"], ["TREND", "NORMAL", "TREND"]):
        run_cycle(broker, "XYZ", window, router)                      # Momentum eligible in both
    assert broker.orders == orders_before
    assert orchestrator.find_position_owner("XYZ") == "Momentum"


def test_owner_signal_exit_still_works_during_cooldown(healthy_log, broker, router):
    run_cycle(broker, "XYZ", ["TREND"] * 3, router)                   # Donchian long
    broker.prices["XYZ"] = 150.0
    orchestrator.orchestrate_one_cycle(                               # VOL_BREAKOUT x1, Donchian eligible, signal 0
        None, "XYZ", broker, StaticDataSource(trend_frame("collapse")),
        router=router, detector=FixedRegimeDetector(history=["TREND", "TREND", "VOL_BREAKOUT"]),
    )
    rows = symbol_rows("XYZ")
    assert rows.iloc[-1]["side"] == "SELL" and rows.iloc[-1]["reason"] == orchestrator.REASON_SIGNAL


def test_unattributed_position_is_flattened_during_cooldown_and_nobody_reenters(healthy_log, broker, router):
    broker.place_order("XYZ", "BUY", 10)
    run_cycle(broker, "XYZ", ["RANGE", "RANGE", "TREND"], router)   # TREND x1: Donchian would otherwise enter
    assert broker.orders[-1] == ("XYZ", "SELL", 10.0)
    assert broker.get_positions() == {}


def test_broken_history_blocks_entries_but_latest_regime_still_protects(healthy_log, broker, router):
    class HistoryBroken(FixedRegimeDetector):
        def regime_history(self, df):
            raise RuntimeError("no history")

    data = StaticDataSource(trend_frame("breakout"))
    orchestrator.orchestrate_one_cycle(None, "XYZ", broker, data, router=router, detector=HistoryBroken("TREND"))
    assert broker.orders == []                                        # unknown stability -> no entry


# ---------------------------------------------------------------------------
# Pinned strategies
# ---------------------------------------------------------------------------
def test_pinned_registered_strategy_respects_cooldown(healthy_log, broker):
    pinned = get_strategy_class("Donchian_Breakout")()
    data = StaticDataSource(trend_frame("breakout"))
    orchestrator.orchestrate_one_cycle(pinned, "XYZ", broker, data,
                                       detector=FixedRegimeDetector(history=["RANGE", "RANGE", "TREND"]))
    assert broker.orders == []
    orchestrator.orchestrate_one_cycle(pinned, "XYZ", broker, data,
                                       detector=FixedRegimeDetector(history=["TREND"] * 3))
    assert len(broker.buys()) == 1


def test_pinned_unregistered_strategy_ignores_cooldown(healthy_log, broker):
    from tests.unit.test_orchestrator import FixedSignalStrategy

    seed_healthy_history("Fixed")
    data = StaticDataSource(trend_frame("breakout"))
    orchestrator.orchestrate_one_cycle(FixedSignalStrategy(1, "Fixed"), "XYZ", broker, data,
                                       detector=FixedRegimeDetector(history=["RANGE", "RANGE", "TREND"]))
    assert len(broker.buys()) == 1


# ---------------------------------------------------------------------------
# Real detector end-to-end: a breakout bar flips TREND -> VOL_BREAKOUT
# ---------------------------------------------------------------------------
def test_real_detector_breakout_bar_is_same_family_and_enters_without_lag(healthy_log, broker, router):
    """With the real RegimeDetector the breakout bar is labelled VOL_BREAKOUT
    after a run of TREND bars.  Both are DIRECTIONAL, so the cooldown does not
    treat the breakout as a new regime and Donchian buys the breakout bar."""
    data = StaticDataSource(trend_frame("breakout"))
    from regimes.detector import RegimeDetector
    tail = list(RegimeDetector().regime_history(trend_frame("breakout")).tail(3))
    assert tail == ["TREND", "TREND", "VOL_BREAKOUT"]          # precondition: label change on the signal bar

    orchestrator.orchestrate_one_cycle(None, "XYZ", broker, data, router=router)
    assert broker.buys() == [("XYZ", "BUY", float(LOT))]
    assert orchestrator.find_position_owner("XYZ") == "Donchian_Breakout"


# ---------------------------------------------------------------------------
# Directional family: TREND <-> VOL_BREAKOUT is not a regime change
# ---------------------------------------------------------------------------
from regimes.stability import RegimeCooldown  # noqa: E402


def test_breakout_bar_after_trend_is_bought_on_the_same_day(healthy_log, broker, router, capsys):
    run_cycle(broker, "XYZ", ["TREND", "TREND", "VOL_BREAKOUT"], router)
    assert broker.buys() == [("XYZ", "BUY", float(LOT))]
    assert orchestrator.find_position_owner("XYZ") == "Donchian_Breakout"
    assert "VOL_BREAKOUT [DIRECTIONAL] x3/3 (stable)" in capsys.readouterr().out


def test_same_day_reallocation_is_not_blocked_within_the_family(healthy_log, broker, router):
    """Momentum (TREND-only) holds; the breakout bar makes it ineligible.  The
    flatten happens and Donchian takes over in the same cycle because the
    directional family is still stable -- no cooldown lag."""
    run_cycle(broker, "XYZ", ["TREND"] * 3, router)
    # router prefers Donchian in TREND; force a Momentum holder via the log + broker
    broker.place_order("XYZ", "SELL", LOT)
    orchestrator.append_trade_to_history(
        {"timestamp": pd.Timestamp.now(), "strategy": "Donchian_Breakout", "symbol": "XYZ", "side": "SELL",
         "qty": LOT, "price": PRICE, "entry_price": PRICE, "fees": 0.0, "pnl": 0.0, "reason": "signal"}
    )
    broker.place_order("XYZ", "BUY", LOT)
    orchestrator.append_trade_to_history(
        {"timestamp": pd.Timestamp.now(), "strategy": "Momentum", "symbol": "XYZ", "side": "BUY",
         "qty": LOT, "price": PRICE, "entry_price": PRICE, "fees": 0.0, "pnl": float("nan"), "reason": "signal"}
    )
    assert orchestrator.find_position_owner("XYZ") == "Momentum"

    run_cycle(broker, "XYZ", ["TREND", "TREND", "VOL_BREAKOUT"], router)
    assert broker.orders[-2:] == [("XYZ", "SELL", float(LOT)), ("XYZ", "BUY", float(LOT))]
    rows = symbol_rows("XYZ").tail(2)
    assert list(zip(rows["strategy"], rows["side"])) == [("Momentum", "SELL"), ("Donchian_Breakout", "BUY")]
    assert rows.iloc[0]["reason"] == f"{orchestrator.REASON_REGIME_SHIFT}:VOL_BREAKOUT"


def test_cross_family_shift_on_the_same_setup_is_still_blocked(healthy_log, broker, router):
    """Control for the test above: NORMAL -> VOL_BREAKOUT is a real shift, so the
    same Momentum holder is flattened but Donchian must wait."""
    run_cycle(broker, "XYZ", ["NORMAL"] * 3, router)                  # Momentum enters
    run_cycle(broker, "XYZ", ["NORMAL", "NORMAL", "VOL_BREAKOUT"], router)
    assert broker.orders[-1] == ("XYZ", "SELL", float(LOT))
    assert broker.get_positions() == {}


def test_directional_flicker_never_blocks_entries_or_resets_the_clock(healthy_log, broker, router):
    """TREND/VOL_BREAKOUT alternating every day: Donchian enters on day 1 and
    holds throughout -- zero churn, zero cooldown."""
    window = ["TREND"] * 3
    run_cycle(broker, "XYZ", window, router)
    for label in ["VOL_BREAKOUT", "TREND", "VOL_BREAKOUT", "TREND", "VOL_BREAKOUT"]:
        window = (window + [label])[-3:]
        run_cycle(broker, "XYZ", window, router)
    assert broker.orders == [("XYZ", "BUY", float(LOT))]
    assert orchestrator.find_position_owner("XYZ") == "Donchian_Breakout"


def test_family_streak_counts_mixed_directional_bars_after_a_real_shift(healthy_log, broker, router):
    run_cycle(broker, "XYZ", ["RANGE", "RANGE", "TREND"], router)          # directional x1 -> wait
    run_cycle(broker, "XYZ", ["RANGE", "TREND", "VOL_BREAKOUT"], router)   # x2 -> wait (VOL bar counts)
    assert broker.orders == []
    run_cycle(broker, "XYZ", ["TREND", "VOL_BREAKOUT", "TREND"], router)   # x3 -> go
    assert broker.buys() == [("XYZ", "BUY", float(LOT))]


def test_leaving_the_directional_family_still_triggers_cooldown_and_flatten(healthy_log, broker, router):
    run_cycle(broker, "XYZ", ["TREND", "VOL_BREAKOUT", "TREND"], router)   # Donchian long
    run_cycle(broker, "XYZ", ["VOL_BREAKOUT", "TREND", "RANGE"], router)   # RANGE x1: flatten, RSI must wait
    assert broker.orders[-1] == ("XYZ", "SELL", float(LOT))
    assert broker.get_positions() == {}
    assert symbol_rows("XYZ").iloc[-1]["reason"] == f"{orchestrator.REASON_REGIME_SHIFT}:RANGE"


def test_custom_cooldown_object_overrides_default_families(healthy_log, broker, router):
    strict = RegimeCooldown(3, families={})                              # label-level: breakout is a new regime
    run_cycle(broker, "XYZ", ["TREND", "TREND", "VOL_BREAKOUT"], router, cooldown=strict)
    assert broker.orders == []
    run_cycle(broker, "XYZ", ["TREND", "TREND", "VOL_BREAKOUT"], router)  # default families: enters
    assert len(broker.buys()) == 1