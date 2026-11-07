import math

import pytest

from core.strategies import REGIME_STRATEGY_MAP, RoutingDecision, StrategyRouter
from core.strategies.donchian_breakout import DonchianBreakout
from core.strategies.momentum import MomentumStrategy
from core.strategies.rsi_mean_reversion import RSIMeanReversion


# ---------------------------------------------------------------------------
# Default policy == registry table
# ---------------------------------------------------------------------------
def test_default_routing_table_mirrors_registry():
    router = StrategyRouter()
    assert router.regime_map == {r: tuple(names) for r, names in REGIME_STRATEGY_MAP.items()}
    assert set(router.known_regimes) == set(REGIME_STRATEGY_MAP)
    assert router.fallback_name is None


@pytest.mark.parametrize(
    "regime, expected_cls",
    [
        ("TREND", DonchianBreakout),  # first candidate wins
        ("VOL_BREAKOUT", DonchianBreakout),
        ("NORMAL", MomentumStrategy),
        ("RANGE", RSIMeanReversion),
    ],
)
def test_known_regimes_route_to_expected_strategy(regime, expected_cls):
    decision = StrategyRouter().route({"regime": regime, "adx": 25.0})
    assert isinstance(decision, RoutingDecision)
    assert decision.regime == regime
    assert isinstance(decision.strategy, expected_cls)
    assert not decision.is_cash
    assert not decision.fallback_used


def test_route_accepts_bare_string_and_normalises_label():
    router = StrategyRouter()
    for raw in ("trend", " TREND ", "Trend"):
        d = router.route(raw)
        assert d.regime == "TREND"
        assert isinstance(d.strategy, DonchianBreakout)


def test_trend_exposes_both_candidates_in_priority_order():
    d = StrategyRouter().route("TREND")
    assert d.candidates == ("Donchian_Breakout", "Momentum")
    assert d.strategy_name == "Donchian_Breakout"
    assert d.reason == "priority"


# ---------------------------------------------------------------------------
# Fallback / cash handling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bad_input",
    ["QUIET", "SOMETHING_NEW", None, "", "   ", {}, {"regime": None}, {"regime": float("nan")}, math.nan],
)
def test_unknown_or_missing_regime_routes_to_cash_by_default(bad_input):
    d = StrategyRouter().route(bad_input)
    assert d.is_cash
    assert d.strategy is None
    assert d.strategy_name is None
    assert d.fallback_used
    assert "cash" in d.reason


def test_unknown_regime_label_is_preserved_in_decision():
    d = StrategyRouter().route("QUIET")
    assert d.regime == "QUIET"
    assert d.candidates == ()


def test_explicit_fallback_strategy_is_used_for_unknown_regimes():
    router = StrategyRouter(fallback="Momentum")
    d = router.route("QUIET")
    assert not d.is_cash
    assert isinstance(d.strategy, MomentumStrategy)
    assert d.fallback_used
    assert router.fallback_name == "Momentum"
    # known regimes are unaffected by the fallback
    assert isinstance(router.route("RANGE").strategy, RSIMeanReversion)


def test_regime_with_empty_candidate_list_falls_back():
    d = StrategyRouter({"TREND": []}).route("TREND")
    assert d.is_cash and d.fallback_used


# ---------------------------------------------------------------------------
# Instance management
# ---------------------------------------------------------------------------
def test_strategy_instances_are_cached_and_shared_across_calls():
    router = StrategyRouter()
    a = router.route("TREND").strategy
    b = router.route("VOL_BREAKOUT").strategy
    assert a is b is router.get_strategy("Donchian_Breakout")
    assert router.has_strategy("Donchian_Breakout")
    assert not router.has_strategy("RSI_MeanReversion")  # lazily built


def test_get_strategy_unknown_name_raises_keyerror():
    with pytest.raises(KeyError):
        StrategyRouter().get_strategy("NoSuchStrategy")


def test_custom_map_accepts_names_classes_and_instances():
    momentum = MomentumStrategy(roc_period=5)
    router = StrategyRouter({"trend": [MomentumStrategy], "range": ["RSI_MeanReversion"], "normal": [momentum]})
    assert router.regime_map == {"TREND": ("Momentum",), "RANGE": ("RSI_MeanReversion",), "NORMAL": ("Momentum",)}
    # the pre-built instance is served as-is
    assert router.route("NORMAL").strategy is momentum
    assert router.route("TREND").strategy is momentum  # same name -> same cached instance


def test_typo_in_custom_map_fails_fast_at_construction():
    with pytest.raises(KeyError):
        StrategyRouter({"TREND": ["Momentm"]})


def test_invalid_spec_type_rejected():
    with pytest.raises(TypeError):
        StrategyRouter({"TREND": [42]})


def test_invalid_regime_key_rejected():
    with pytest.raises(ValueError):
        StrategyRouter({None: ["Momentum"]})


# ---------------------------------------------------------------------------
# select() hook
# ---------------------------------------------------------------------------
def test_select_override_can_change_priority():
    class PreferMomentum(StrategyRouter):
        def select(self, regime, candidates):
            return "Momentum" if "Momentum" in candidates else candidates[0]

    d = PreferMomentum().route("TREND")
    assert isinstance(d.strategy, MomentumStrategy)


def test_select_returning_none_declines_to_cash():
    class Decline(StrategyRouter):
        def select(self, regime, candidates):
            return None

    d = Decline().route("TREND")
    assert d.is_cash and d.fallback_used and d.candidates == ("Donchian_Breakout", "Momentum")


def test_select_returning_non_candidate_is_an_error():
    class Rogue(StrategyRouter):
        def select(self, regime, candidates):
            return "RSI_MeanReversion"

    with pytest.raises(ValueError):
        Rogue().route("TREND")


def test_candidates_for_unknown_regime_is_empty():
    router = StrategyRouter()
    assert router.candidates_for("QUIET") == []
    assert router.candidates_for(None) == []
    assert router.candidates_for("range") == ["RSI_MeanReversion"]


def test_decision_str_is_log_friendly():
    assert "Donchian_Breakout" in str(StrategyRouter().route("TREND"))
    assert "CASH" in str(StrategyRouter().route("QUIET"))


def test_explicit_instance_wins_over_class_regardless_of_map_order():
    momentum = MomentumStrategy(roc_period=5)
    router = StrategyRouter({"normal": [momentum], "trend": [MomentumStrategy]})
    assert router.route("TREND").strategy is momentum
    router = StrategyRouter({"trend": [MomentumStrategy], "normal": [momentum]})
    assert router.route("TREND").strategy is momentum


def test_two_distinct_instances_with_same_name_are_rejected():
    with pytest.raises(ValueError):
        StrategyRouter({"trend": [MomentumStrategy(roc_period=5)], "normal": [MomentumStrategy(roc_period=9)]})
    # the same instance listed twice is fine
    m = MomentumStrategy()
    assert StrategyRouter({"trend": [m], "normal": [m]}).route("NORMAL").strategy is m