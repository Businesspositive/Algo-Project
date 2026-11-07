"""
Unit tests for the pieces the regime-transition mechanism is built from:
PendingLiquidation / SettlementLedger, the settlement-aware exposure maths,
trade-log position ownership, and the cycle planner.
"""
from datetime import timedelta

import pandas as pd
import pytest

import orchestrator
from core.strategies import REGIME_STRATEGY_MAP, StrategyRouter
from core.strategies.registry import get_strategy_class
from tests.support import ScriptedBroker


# ---------------------------------------------------------------------------
# PendingLiquidation
# ---------------------------------------------------------------------------
def test_unreflected_qty_depends_on_what_broker_still_shows():
    liq = orchestrator.PendingLiquidation("AAA", qty=30, price=10.0, proceeds=300.0, pre_qty=50)
    assert liq.unreflected_qty(shown_qty=50) == 0      # broker still shows full pre-sale holding
    assert liq.unreflected_qty(shown_qty=20) == 30     # broker already removed the sold shares
    assert liq.unreflected_qty(shown_qty=35) == 15     # fill only partly propagated
    assert liq.unreflected_qty(shown_qty=0) == 30      # never more than what was sold
    assert liq.unreflected_qty(shown_qty=None) == 30


def test_pending_liquidation_basics():
    liq = orchestrator.PendingLiquidation("AAA", qty=10, price=12.5, proceeds=-1.0, pre_qty=10, reason="x")
    assert liq.notional == 125.0
    assert liq.proceeds == 0.0  # never negative
    assert isinstance(liq.timestamp, pd.Timestamp)
    assert "AAA" in repr(liq)


# ---------------------------------------------------------------------------
# SettlementLedger
# ---------------------------------------------------------------------------
def make_liq(symbol="AAA", qty=10, price=10.0, age_hours=0.0):
    return orchestrator.PendingLiquidation(
        symbol, qty, price, proceeds=qty * price, pre_qty=qty,
        timestamp=pd.Timestamp.now() - pd.Timedelta(hours=age_hours),
    )


def test_ledger_records_and_totals():
    ledger = orchestrator.SettlementLedger()
    assert len(ledger) == 0 and ledger.unsettled_proceeds == 0.0
    ledger.record(make_liq("AAA", 10, 10.0))
    ledger.record(make_liq("BBB", 5, 20.0))
    ledger.record(None)                                # ignored
    ledger.record(make_liq("CCC", 0, 99.0))            # zero qty ignored
    assert len(ledger) == 2
    assert ledger.unsettled_proceeds == pytest.approx(200.0)


def test_ledger_settle_by_symbol_and_clear():
    ledger = orchestrator.SettlementLedger()
    ledger.record(make_liq("AAA"))
    ledger.record(make_liq("BBB"))
    ledger.settle("AAA")
    assert [x.symbol for x in ledger.pending()] == ["BBB"]
    ledger.clear()
    assert len(ledger) == 0


def test_ledger_without_delay_never_expires_on_its_own():
    ledger = orchestrator.SettlementLedger()
    ledger.record(make_liq(age_hours=24 * 30))
    assert len(ledger) == 1


def test_ledger_with_settlement_delay_expires_old_entries():
    ledger = orchestrator.SettlementLedger(settlement_delay=timedelta(days=1))
    ledger.record(make_liq("FRESH", age_hours=23))
    ledger.record(make_liq("OLD", age_hours=25))
    assert [x.symbol for x in ledger.pending()] == ["FRESH"]
    assert len(ledger) == 1


# ---------------------------------------------------------------------------
# Settlement-aware exposure snapshot
# ---------------------------------------------------------------------------
def test_snapshot_without_pending_matches_legacy_fields():
    broker = ScriptedBroker({"AAA": 100.0})
    broker.place_order("AAA", "BUY", 100)  # 10,000 deployed
    snap = orchestrator.compute_portfolio_exposure(broker)
    assert snap["cash"] == 90_000 and snap["settled_cash"] == 90_000
    assert snap["positions_value"] == pytest.approx(10_000)
    assert snap["in_flight_value"] == 0.0 and snap["unsettled_proceeds"] == 0.0
    assert snap["exposure"] == pytest.approx(0.10)


def test_liquidated_shares_count_as_exposure_until_settled_and_cash_is_withheld():
    broker = ScriptedBroker({"AAA": 100.0})
    broker.place_order("AAA", "BUY", 100)
    broker.place_order("AAA", "SELL", 100)  # paper broker: position gone, cash back to 100,000
    liq = orchestrator.PendingLiquidation("AAA", qty=100, price=100.0, proceeds=10_000, pre_qty=100)

    snap = orchestrator.compute_portfolio_exposure(broker, pending=liq)
    assert snap["positions_value"] == 0.0
    assert snap["in_flight_value"] == pytest.approx(10_000)
    assert snap["exposure"] == pytest.approx(0.10)          # still 10% deployed from the gate's point of view
    assert snap["cash"] == 100_000
    assert snap["unsettled_proceeds"] == 10_000
    assert snap["settled_cash"] == 90_000                   # proceeds not spendable


def test_no_double_count_when_broker_still_shows_sold_position():
    """A real broker may not have propagated the fill yet: the shares are
    already in positions_value, so in-flight must not add them again."""
    broker = ScriptedBroker({"AAA": 100.0})
    broker.place_order("AAA", "BUY", 100)
    liq = orchestrator.PendingLiquidation("AAA", qty=100, price=100.0, proceeds=10_000, pre_qty=100)

    snap = orchestrator.compute_portfolio_exposure(broker, pending=liq)
    assert snap["positions_value"] == pytest.approx(10_000)
    assert snap["in_flight_value"] == 0.0
    assert snap["exposure"] == pytest.approx(0.10)
    # cash is still withheld -- conservative, never inflates the next order
    assert snap["settled_cash"] == pytest.approx(90_000 - 10_000)


def test_settled_cash_never_goes_negative():
    broker = ScriptedBroker({"AAA": 100.0}, cash=1_000.0)
    liq = orchestrator.PendingLiquidation("AAA", qty=100, price=100.0, proceeds=10_000, pre_qty=100)
    assert orchestrator.compute_portfolio_exposure(broker, pending=liq)["settled_cash"] == 0.0


def test_snapshot_accepts_ledger_single_or_list():
    broker = ScriptedBroker({"AAA": 100.0, "BBB": 50.0})
    liq_a = orchestrator.PendingLiquidation("AAA", 10, 100.0, 1_000, pre_qty=10)
    liq_b = orchestrator.PendingLiquidation("BBB", 10, 50.0, 500, pre_qty=10)
    ledger = orchestrator.SettlementLedger()
    ledger.record(liq_a)
    ledger.record(liq_b)
    for pending in (ledger, [liq_a, liq_b], (liq_a, liq_b)):
        snap = orchestrator.compute_portfolio_exposure(broker, pending=pending)
        assert snap["in_flight_value"] == pytest.approx(1_500)
        assert snap["unsettled_proceeds"] == pytest.approx(1_500)
    assert orchestrator.compute_portfolio_exposure(broker, pending=liq_a)["in_flight_value"] == pytest.approx(1_000)


def test_in_flight_valued_at_lookup_price_when_available():
    broker = ScriptedBroker({"AAA": 100.0})
    liq = orchestrator.PendingLiquidation("AAA", 10, price=100.0, proceeds=1_000, pre_qty=10)
    snap = orchestrator.compute_portfolio_exposure(broker, {"AAA": 120.0}, pending=liq)
    assert snap["in_flight_value"] == pytest.approx(1_200)


# ---------------------------------------------------------------------------
# Total-exposure gate with in-flight value
# ---------------------------------------------------------------------------
def test_total_exposure_gate_includes_in_flight_value_at_25pct_boundary():
    broker = ScriptedBroker({"AAA": 100.0, "XYZ": 10.0})
    broker.place_order("AAA", "BUY", 200)   # 20% deployed, 80,000 cash
    broker.place_order("AAA", "SELL", 50)   # liquidate a quarter -> broker shows 15%
    liq = orchestrator.PendingLiquidation("AAA", qty=50, price=100.0, proceeds=5_000, pre_qty=200)

    # without settlement awareness the book looks 15% deployed and a 10% order passes
    ok, d = orchestrator.check_total_exposure(broker, "XYZ", qty=1_000, price=10.0)
    assert ok and d["projected_exposure"] == pytest.approx(0.25)

    # with the unsettled 5% in flight, the same order would be 30% -> blocked
    ok, d = orchestrator.check_total_exposure(broker, "XYZ", qty=1_000, price=10.0, pending=liq)
    assert not ok
    assert d["in_flight_value"] == pytest.approx(5_000)
    assert d["projected_exposure"] == pytest.approx(0.30)

    # exactly at the limit is still allowed: 15% + 5% in flight + 5% new = 25%
    ok, d = orchestrator.check_total_exposure(broker, "XYZ", qty=500, price=10.0, pending=liq)
    assert ok and d["projected_exposure"] == pytest.approx(0.25)
    # one share over is not
    ok, _ = orchestrator.check_total_exposure(broker, "XYZ", qty=501, price=10.0, pending=liq)
    assert not ok


def test_total_exposure_gate_uses_in_flight_from_precomputed_snapshot():
    broker = ScriptedBroker({"AAA": 100.0})
    liq = orchestrator.PendingLiquidation("AAA", qty=200, price=100.0, proceeds=20_000, pre_qty=200)
    snap = orchestrator.compute_portfolio_exposure(broker, pending=liq)
    ok, d = orchestrator.check_total_exposure(broker, "XYZ", qty=600, price=10.0, snapshot=snap)
    assert not ok and d["projected_exposure"] == pytest.approx(0.26)


def test_legacy_snapshot_without_in_flight_key_still_works():
    broker = ScriptedBroker({"AAA": 100.0})
    snap = {"cash": 100_000.0, "positions_value": 0.0, "portfolio_value": 100_000.0, "exposure": 0.0}
    ok, d = orchestrator.check_total_exposure(broker, "XYZ", qty=100, price=10.0, snapshot=snap)
    assert ok and d["projected_exposure"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Registry-derived regime gate
# ---------------------------------------------------------------------------
def test_regime_map_is_derived_from_registry():
    for regime, names in REGIME_STRATEGY_MAP.items():
        for name in names:
            assert regime in orchestrator.REGIME_MAP[name]["allowed"]
    assert "Fixed" not in orchestrator.REGIME_MAP


@pytest.mark.parametrize(
    "name, regime, routed, expected",
    [
        ("Donchian_Breakout", "TREND", True, True),
        ("Donchian_Breakout", "RANGE", True, False),
        ("Momentum", "TREND", True, True),       # second candidate still eligible
        ("Momentum", "VOL_BREAKOUT", True, False),
        ("RSI_MeanReversion", "QUIET", True, False),
        ("RSI_MeanReversion", None, True, False),  # unknown regime: nothing is eligible
        ("Donchian_Breakout", "TREND", False, True),
        ("Donchian_Breakout", "RANGE", False, False),
        ("Fixed", "QUIET", False, True),           # unregistered pinned strategy: unrestricted
        ("Fixed", None, False, True),
    ],
)
def test_strategy_eligible_in_regime(name, regime, routed, expected):
    router = StrategyRouter() if routed else None
    assert orchestrator.strategy_eligible_in_regime(name, regime, router) is expected


# ---------------------------------------------------------------------------
# Position ownership from the trade log
# ---------------------------------------------------------------------------
def log(strategy, symbol, side, qty, ts, price=100.0, **extra):
    orchestrator.append_trade_to_history(
        {"timestamp": ts, "strategy": strategy, "symbol": symbol, "side": side, "qty": qty,
         "price": price, "entry_price": price, "fees": 0.0, "pnl": float("nan"), **extra}
    )


def test_find_position_owner_none_when_no_log_or_closed(paper_trade_log):
    assert orchestrator.find_position_owner("XYZ") is None
    t0 = pd.Timestamp("2025-01-01")
    log("Momentum", "XYZ", "BUY", 10, t0)
    assert orchestrator.find_position_owner("XYZ") == "Momentum"
    log("Momentum", "XYZ", "SELL", 10, t0 + pd.Timedelta(days=1))
    assert orchestrator.find_position_owner("XYZ") is None


def test_find_position_owner_survives_partial_exit_and_picks_latest_opener(paper_trade_log):
    t0 = pd.Timestamp("2025-01-01")
    log("Momentum", "XYZ", "BUY", 10, t0)
    log("Momentum", "XYZ", "SELL", 4, t0 + pd.Timedelta(days=1))   # partial exit: still open
    assert orchestrator.find_position_owner("XYZ") == "Momentum"
    # a later BUY by another strategy wins
    log("Donchian_Breakout", "XYZ", "BUY", 5, t0 + pd.Timedelta(days=2))
    assert orchestrator.find_position_owner("XYZ") == "Donchian_Breakout"
    # other symbols do not interfere
    log("RSI_MeanReversion", "ABC", "BUY", 5, t0 + pd.Timedelta(days=3))
    assert orchestrator.find_position_owner("XYZ") == "Donchian_Breakout"
    assert orchestrator.find_position_owner("ABC") == "RSI_MeanReversion"


# ---------------------------------------------------------------------------
# plan_cycle: the transition policy, independent of broker / data
# ---------------------------------------------------------------------------
@pytest.fixture
def router():
    return StrategyRouter()


def open_position_owned_by(name, symbol="XYZ"):
    log(name, symbol, "BUY", 21, pd.Timestamp("2025-01-01"))


def test_plan_flat_routed_and_cash(router):
    plan = orchestrator.plan_cycle(router.route("TREND"), "XYZ", in_position=False, router=router)
    assert plan.strategy.name == "Donchian_Breakout" and plan.flatten is None

    plan = orchestrator.plan_cycle(router.route("QUIET"), "XYZ", in_position=False, router=router)
    assert plan.strategy is None and plan.flatten is None


def test_plan_flat_pinned_ignores_router(router):
    pinned = get_strategy_class("RSI_MeanReversion")()
    plan = orchestrator.plan_cycle(router.route("TREND"), "XYZ", False, router, pinned=pinned)
    assert plan.strategy is pinned and plan.flatten is None


def test_plan_long_owner_still_eligible_keeps_position(router, paper_trade_log):
    open_position_owned_by("Momentum")
    plan = orchestrator.plan_cycle(router.route("TREND"), "XYZ", True, router)  # TREND lists Momentum 2nd
    assert plan.flatten is None
    assert plan.strategy is router.get_strategy("Momentum")


def test_plan_long_hard_shift_flattens_owner_then_hands_over(router, paper_trade_log):
    open_position_owned_by("Donchian_Breakout")
    plan = orchestrator.plan_cycle(router.route("RANGE"), "XYZ", True, router)
    assert plan.flatten is router.get_strategy("Donchian_Breakout")
    assert plan.flatten_reason == f"{orchestrator.REASON_REGIME_SHIFT}:RANGE"
    assert plan.strategy is router.get_strategy("RSI_MeanReversion")


@pytest.mark.parametrize("regime_input, label", [("QUIET", "QUIET"), (None, "None"), ({"regime": "weird"}, "WEIRD")])
def test_plan_long_cash_regime_flattens_and_runs_nothing(router, paper_trade_log, regime_input, label):
    open_position_owned_by("Donchian_Breakout")
    plan = orchestrator.plan_cycle(router.route(regime_input), "XYZ", True, router)
    assert plan.flatten is router.get_strategy("Donchian_Breakout")
    assert plan.flatten_reason == f"{orchestrator.REASON_REGIME_CASH}:{label}"
    assert plan.strategy is None


def test_plan_long_unattributed_position_is_flattened_under_unattributed(router, paper_trade_log):
    plan = orchestrator.plan_cycle(router.route("TREND"), "XYZ", True, router)
    assert plan.flatten is not None and plan.flatten.name == orchestrator.UNATTRIBUTED_STRATEGY
    assert plan.flatten_reason == orchestrator.REASON_UNATTRIBUTED
    assert plan.strategy is router.get_strategy("Donchian_Breakout")


def test_plan_long_unregistered_owner_treated_as_unattributed(router, paper_trade_log):
    open_position_owned_by("SomeRetiredStrategy")
    plan = orchestrator.plan_cycle(router.route("RANGE"), "XYZ", True, router)
    assert plan.flatten.name == orchestrator.UNATTRIBUTED_STRATEGY


def test_plan_force_flatten_false_restores_permissive_behaviour(router, paper_trade_log):
    open_position_owned_by("Donchian_Breakout")
    plan = orchestrator.plan_cycle(router.route("RANGE"), "XYZ", True, router, force_flatten=False)
    assert plan.flatten is None and plan.strategy is router.get_strategy("Donchian_Breakout")

    # unattributed + permissive: adopt the routed strategy, or hold on cash
    paper_trade_log.unlink()
    plan = orchestrator.plan_cycle(router.route("RANGE"), "XYZ", True, router, force_flatten=False)
    assert plan.flatten is None and plan.strategy is router.get_strategy("RSI_MeanReversion")
    plan = orchestrator.plan_cycle(router.route("QUIET"), "XYZ", True, router, force_flatten=False)
    assert plan.flatten is None and plan.strategy is None


def test_plan_pinned_long_ineligible_flattens_without_reentry(router):
    pinned = get_strategy_class("Donchian_Breakout")()
    plan = orchestrator.plan_cycle(router.route("RANGE"), "XYZ", True, router, pinned=pinned)
    assert plan.flatten is pinned and plan.flatten_reason.startswith(orchestrator.REASON_REGIME_SHIFT)
    assert plan.strategy is None


def test_plan_pinned_long_eligible_keeps_managing(router):
    pinned = get_strategy_class("Donchian_Breakout")()
    plan = orchestrator.plan_cycle(router.route("TREND"), "XYZ", True, router, pinned=pinned)
    assert plan.flatten is None and plan.strategy is pinned


def test_unattributed_owner_always_signals_flat(sample_ohlcv):
    stand_in = orchestrator._UnattributedOwner()
    assert stand_in.name == orchestrator.UNATTRIBUTED_STRATEGY
    assert (stand_in.generate_signals(sample_ohlcv)["signal"] == 0).all()