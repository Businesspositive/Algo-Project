"""
Unit tests for the regime-transition cooldown: the stability assessment
itself, the detector's per-bar history it is computed from, and how the
cycle planner honours it.
"""
import math

import pandas as pd
import pytest

import orchestrator
from core.strategies import StrategyRouter
from core.strategies.registry import get_strategy_class
from core.strategies.base import Strategy
from regimes.detector import RegimeDetector, normalise_regime
from regimes.stability import DEFAULT_MIN_STABLE_PERIODS, RegimeCooldown, RegimeStability


# ---------------------------------------------------------------------------
# normalise_regime (shared by detector, router and cooldown)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [("TREND", "TREND"), (" trend ", "TREND"), ("Range", "RANGE"), ("", None), ("   ", None),
     (None, None), (math.nan, None), (float("nan"), None), (123, "123")],
)
def test_normalise_regime(raw, expected):
    assert normalise_regime(raw) == expected


def test_router_uses_the_shared_normaliser():
    assert StrategyRouter.normalise_regime(" vol_breakout ") == "VOL_BREAKOUT"


# ---------------------------------------------------------------------------
# RegimeCooldown.assess
# ---------------------------------------------------------------------------
def test_default_is_three_periods():
    assert DEFAULT_MIN_STABLE_PERIODS == 3
    assert RegimeCooldown().min_periods == 3


@pytest.mark.parametrize("bad", [0, -1])
def test_min_periods_must_be_positive(bad):
    with pytest.raises(ValueError):
        RegimeCooldown(bad)


def test_long_stable_history_is_stable():
    s = RegimeCooldown(3).assess(["TREND"] * 10)
    assert s == RegimeStability("TREND", 3, 3, None, family="DIRECTIONAL")
    assert RegimeCooldown(3).assess(["RANGE"] * 10) == RegimeStability("RANGE", 3, 3, None)  # own family
    assert s.stable and not s.in_cooldown and s.remaining == 0


@pytest.mark.parametrize(
    "history, streak, previous, remaining",
    [
        (["TREND", "TREND", "RANGE"], 1, "TREND", 2),
        (["TREND", "RANGE", "RANGE"], 2, "TREND", 1),
        (["RANGE", "RANGE", "RANGE"], 3, None, 0),
        (["TREND", "RANGE", "RANGE", "RANGE"], 3, "TREND", 0),          # stable but change still visible
        (["TREND", "RANGE", "TREND", "RANGE", "TREND"], 1, "RANGE", 2),  # flicker: never settles
        (["RANGE"], 1, None, 2),                                          # short window
    ],
)
def test_streak_previous_and_remaining(history, streak, previous, remaining):
    s = RegimeCooldown(3).assess(history)
    assert s.regime == history[-1]
    assert s.streak == streak
    assert s.previous == previous
    assert s.remaining == remaining
    assert s.stable is (remaining == 0)


def test_labels_are_normalised_before_comparison():
    s = RegimeCooldown(3).assess(["trend", " TREND", "Trend"])
    assert s.regime == "TREND" and s.stable


@pytest.mark.parametrize("history", [[], None, ["TREND", "TREND", None], ["TREND", float("nan")]])
def test_missing_or_unknown_latest_label_is_never_stable(history):
    s = RegimeCooldown(3).assess(history)
    assert s.regime is None and s.streak == 0
    assert s.in_cooldown and not s.stable
    assert "unknown" in str(s)


def test_unknown_bar_inside_window_breaks_the_streak():
    s = RegimeCooldown(3).assess(["TREND", None, "TREND", "TREND"])
    assert s.streak == 2 and s.previous is None and s.in_cooldown


def test_min_periods_one_disables_cooldown():
    s = RegimeCooldown(1).assess(["TREND", "RANGE"])
    assert s.stable and s.streak == 1 and s.previous == "TREND"


def test_accepts_pandas_series():
    series = pd.Series(["TREND", "TREND", "RANGE", "RANGE"], index=pd.date_range("2025-01-01", periods=4))
    s = RegimeCooldown(2).assess(series)
    assert s.regime == "RANGE" and s.stable and s.previous == "TREND"


def test_str_is_log_friendly():
    assert str(RegimeCooldown(3).assess(["TREND", "TREND", "RANGE"])) == \
        "RANGE x1/3 (cooldown, 2 more bar(s) needed, was TREND)"
    assert str(RegimeCooldown(3).assess(["RANGE"] * 3)) == "RANGE x3/3 (stable)"


# ---------------------------------------------------------------------------
# RegimeDetector.regime_history
# ---------------------------------------------------------------------------
def test_detector_history_is_aligned_and_consistent_with_latest_regime(sample_ohlcv):
    detector = RegimeDetector()
    history = detector.regime_history(sample_ohlcv)
    assert isinstance(history, pd.Series)
    assert history.index.equals(sample_ohlcv.index)
    assert history.iloc[-1] == detector.latest_regime(sample_ohlcv)["regime"]
    assert set(history.dropna().unique()) <= {"QUIET", "VOL_BREAKOUT", "TREND", "RANGE", "NORMAL"}


def test_orchestrator_regime_history_helper_never_raises():
    class Broken(RegimeDetector):
        def regime_history(self, df):
            raise RuntimeError("boom")

    assert orchestrator.regime_history(pd.DataFrame({"Close": [1.0]}), Broken()) is None
    assert RegimeCooldown().assess(None).in_cooldown


# ---------------------------------------------------------------------------
# plan_cycle under cooldown
# ---------------------------------------------------------------------------
COOLDOWN = RegimeCooldown(3).assess(["TREND", "TREND", "RANGE"])     # RANGE x1/3
STABLE = RegimeCooldown(3).assess(["RANGE", "RANGE", "RANGE"])
COOLDOWN_TREND = RegimeCooldown(3).assess(["RANGE", "RANGE", "TREND"])
COOLDOWN_CASH = RegimeCooldown(3).assess(["TREND", "TREND", "QUIET"])


@pytest.fixture
def router():
    return StrategyRouter()


def open_position_owned_by(name, symbol="XYZ"):
    orchestrator.append_trade_to_history(
        {"timestamp": pd.Timestamp("2025-01-01"), "strategy": name, "symbol": symbol, "side": "BUY",
         "qty": 21, "price": 100.0, "entry_price": 100.0, "fees": 0.0, "pnl": float("nan")}
    )


def test_flat_routed_entry_blocked_during_cooldown(router):
    plan = orchestrator.plan_cycle(router.route("RANGE"), "XYZ", False, router, stability=COOLDOWN)
    assert plan.strategy is None and plan.flatten is None
    assert "cooldown" in plan.reason


def test_flat_routed_entry_allowed_once_stable(router):
    plan = orchestrator.plan_cycle(router.route("RANGE"), "XYZ", False, router, stability=STABLE)
    assert plan.strategy is router.get_strategy("RSI_MeanReversion")


def test_no_stability_given_means_no_cooldown(router):
    plan = orchestrator.plan_cycle(router.route("RANGE"), "XYZ", False, router)
    assert plan.strategy is router.get_strategy("RSI_MeanReversion")


def test_ineligible_owner_is_flattened_immediately_but_nobody_takes_over(router, paper_trade_log):
    open_position_owned_by("Donchian_Breakout")
    plan = orchestrator.plan_cycle(router.route("RANGE"), "XYZ", True, router, stability=COOLDOWN)
    assert plan.flatten is router.get_strategy("Donchian_Breakout")
    assert plan.flatten_reason == f"{orchestrator.REASON_REGIME_SHIFT}:RANGE"
    assert plan.strategy is None
    assert "cooldown" in plan.reason


def test_ineligible_owner_hands_over_once_stable(router, paper_trade_log):
    open_position_owned_by("Donchian_Breakout")
    plan = orchestrator.plan_cycle(router.route("RANGE"), "XYZ", True, router, stability=STABLE)
    assert plan.flatten is router.get_strategy("Donchian_Breakout")
    assert plan.strategy is router.get_strategy("RSI_MeanReversion")


def test_eligible_owner_keeps_position_through_cooldown(router, paper_trade_log):
    """Soft shift: Momentum opened in NORMAL, regime flickers to TREND where it is still a
    candidate -> hold; flattening here would be pure fee churn."""
    open_position_owned_by("Momentum")
    plan = orchestrator.plan_cycle(router.route("TREND"), "XYZ", True, router, stability=COOLDOWN_TREND)
    assert plan.flatten is None and plan.strategy is router.get_strategy("Momentum")


def test_cash_regime_during_cooldown_still_flattens(router, paper_trade_log):
    open_position_owned_by("Donchian_Breakout")
    plan = orchestrator.plan_cycle(router.route("QUIET"), "XYZ", True, router, stability=COOLDOWN_CASH)
    assert plan.flatten is router.get_strategy("Donchian_Breakout")
    assert plan.flatten_reason == f"{orchestrator.REASON_REGIME_CASH}:QUIET"
    assert plan.strategy is None


def test_unattributed_position_flattened_during_cooldown_without_takeover(router, paper_trade_log):
    plan = orchestrator.plan_cycle(router.route("RANGE"), "XYZ", True, router, stability=COOLDOWN)
    assert plan.flatten.name == orchestrator.UNATTRIBUTED_STRATEGY
    assert plan.strategy is None and "cooldown" in plan.reason


def test_pinned_registered_strategy_blocked_during_cooldown(router):
    pinned = get_strategy_class("RSI_MeanReversion")()
    plan = orchestrator.plan_cycle(router.route("RANGE"), "XYZ", False, router, pinned=pinned, stability=COOLDOWN)
    assert plan.strategy is None and "cooldown" in plan.reason


def test_pinned_unregistered_strategy_is_not_subject_to_cooldown(router):
    class Fixed(Strategy):
        def __init__(self):
            super().__init__("Fixed")

        def generate_signals(self, df):
            return df

    pinned = Fixed()
    plan = orchestrator.plan_cycle(router.route("RANGE"), "XYZ", False, router, pinned=pinned, stability=COOLDOWN)
    assert plan.strategy is pinned


# ---------------------------------------------------------------------------
# Regime families: TREND and VOL_BREAKOUT share one streak
# ---------------------------------------------------------------------------
from regimes.stability import DEFAULT_REGIME_FAMILIES  # noqa: E402


def test_default_families_group_trend_and_vol_breakout():
    assert DEFAULT_REGIME_FAMILIES == {"DIRECTIONAL": ("TREND", "VOL_BREAKOUT")}
    c = RegimeCooldown()
    assert c.family_of("TREND") == c.family_of("VOL_BREAKOUT") == "DIRECTIONAL"
    assert c.family_of(" trend ") == "DIRECTIONAL"
    for own in ("RANGE", "NORMAL", "QUIET", "SOMETHING_NEW"):
        assert c.family_of(own) == own          # ungrouped labels are their own family
    assert c.family_of(None) is None and c.family_of(float("nan")) is None
    assert c.same_family("TREND", "VOL_BREAKOUT")
    assert not c.same_family("TREND", "RANGE")
    assert not c.same_family(None, None)


@pytest.mark.parametrize(
    "history",
    [
        ["TREND", "TREND", "VOL_BREAKOUT"],
        ["VOL_BREAKOUT", "VOL_BREAKOUT", "TREND"],
        ["TREND", "VOL_BREAKOUT", "TREND"],
        ["VOL_BREAKOUT", "TREND", "VOL_BREAKOUT"],
        ["RANGE", "TREND", "VOL_BREAKOUT", "TREND"],     # change before the window of 3
    ],
)
def test_switching_within_directional_family_does_not_reset_the_streak(history):
    s = RegimeCooldown(3).assess(history)
    assert s.regime == history[-1]
    assert s.family == "DIRECTIONAL"
    assert s.streak == 3 and s.stable and s.remaining == 0


def test_family_streak_accumulates_across_mixed_labels_after_a_real_shift():
    c = RegimeCooldown(3)
    day1 = c.assess(["RANGE", "RANGE", "TREND"])
    day2 = c.assess(["RANGE", "TREND", "VOL_BREAKOUT"])
    day3 = c.assess(["TREND", "VOL_BREAKOUT", "TREND"])
    assert (day1.streak, day1.stable, day1.previous) == (1, False, "RANGE")
    assert (day2.streak, day2.stable, day2.previous) == (2, False, "RANGE")   # VOL bar continues the count
    assert (day3.streak, day3.stable) == (3, True)


@pytest.mark.parametrize(
    "history, previous",
    [
        (["TREND", "TREND", "RANGE"], "TREND"),
        (["VOL_BREAKOUT", "VOL_BREAKOUT", "RANGE"], "VOL_BREAKOUT"),
        (["RANGE", "RANGE", "VOL_BREAKOUT"], "RANGE"),
        (["TREND", "TREND", "NORMAL"], "TREND"),
        (["TREND", "TREND", "QUIET"], "TREND"),
        (["NORMAL", "NORMAL", "TREND"], "NORMAL"),
    ],
)
def test_shift_between_families_still_triggers_cooldown(history, previous):
    s = RegimeCooldown(3).assess(history)
    assert s.streak == 1 and s.in_cooldown and s.remaining == 2
    assert s.previous == previous


def test_str_shows_family_when_it_differs_from_label():
    assert str(RegimeCooldown(3).assess(["TREND", "TREND", "VOL_BREAKOUT"])) == \
        "VOL_BREAKOUT [DIRECTIONAL] x3/3 (stable)"
    assert str(RegimeCooldown(3).assess(["RANGE", "RANGE", "TREND"])) == \
        "TREND [DIRECTIONAL] x1/3 (cooldown, 2 more bar(s) needed, was RANGE)"
    assert str(RegimeCooldown(3).assess(["RANGE"] * 3)) == "RANGE x3/3 (stable)"


def test_empty_families_restores_strict_label_level_cooldown():
    strict = RegimeCooldown(3, families={})
    s = strict.assess(["TREND", "TREND", "VOL_BREAKOUT"])
    assert s.streak == 1 and s.in_cooldown and s.previous == "TREND" and s.family == "VOL_BREAKOUT"
    assert strict.families == {}


def test_custom_families_are_honoured_and_normalised():
    c = RegimeCooldown(3, families={"calm": ["range", " Normal "], "directional": ["TREND", "VOL_BREAKOUT"]})
    assert c.families == {"CALM": ("RANGE", "NORMAL"), "DIRECTIONAL": ("TREND", "VOL_BREAKOUT")}
    assert c.assess(["RANGE", "NORMAL", "RANGE"]).stable
    assert c.assess(["RANGE", "NORMAL", "TREND"]).in_cooldown


@pytest.mark.parametrize(
    "families",
    [
        {"A": ["TREND"], "B": ["TREND"]},     # label in two families
        {"": ["TREND"]},                      # blank family name
        {"A": ["TREND", None]},               # invalid label
    ],
)
def test_invalid_family_configuration_is_rejected(families):
    with pytest.raises(ValueError):
        RegimeCooldown(3, families=families)


def test_plan_cycle_allows_entry_on_breakout_bar_after_trend(router):
    breakout_bar = RegimeCooldown(3).assess(["TREND", "TREND", "VOL_BREAKOUT"])
    plan = orchestrator.plan_cycle(router.route("VOL_BREAKOUT"), "XYZ", False, router, stability=breakout_bar)
    assert plan.strategy is router.get_strategy("Donchian_Breakout")
    assert "cooldown" not in plan.reason