"""
Regime -> strategy routing.

`RegimeDetector.latest_regime()` tells us *what the market is doing*; this
module decides *which strategy should act on it*.  It is deliberately a pure
policy object with no I/O so the orchestrator (or a backtester) can call it
every cycle and log the decision.

The default policy is the registry's `REGIME_STRATEGY_MAP`:

    RANGE        -> RSI_MeanReversion
    VOL_BREAKOUT -> Donchian_Breakout
    TREND        -> Donchian_Breakout, Momentum   (first entry wins)
    NORMAL       -> Momentum

Any regime that is not in the map (e.g. "QUIET", which the detector emits for
markets too quiet to trade, or a label from a future detector version) routes
to the *fallback*.  By default the fallback is "no strategy", which the caller
should interpret as "stay in / go to cash".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, Union

from regimes.detector import normalise_regime

from .base import Strategy
from .registry import REGIME_STRATEGY_MAP, get_strategy_class

# What a regime map entry may contain: a registry name, a Strategy subclass or
# an already-built instance (handy for injecting ad-hoc strategies).
StrategySpec = Union[str, Type[Strategy], Strategy]


@dataclass(frozen=True)
class RoutingDecision:
    """
    Outcome of routing one regime observation.

    `strategy is None` means "do not trade this cycle" (cash-only). `reason` is
    a short human-readable explanation intended for logs.
    """

    regime: Optional[str]
    strategy: Optional[Strategy]
    reason: str
    candidates: Tuple[str, ...] = ()
    fallback_used: bool = False

    @property
    def is_cash(self) -> bool:
        return self.strategy is None

    @property
    def strategy_name(self) -> Optional[str]:
        return self.strategy.name if self.strategy is not None else None

    def __str__(self) -> str:
        target = self.strategy_name or "CASH"
        return f"regime={self.regime!r} -> {target} ({self.reason})"


class StrategyRouter:
    """
    Map a regime label (or the dict returned by `RegimeDetector.latest_regime`)
    to a concrete `Strategy` instance.

    Strategy instances are built lazily and cached, so a router used across
    many cycles hands back the same object for the same strategy name.  Cached
    instances are keyed by the *registry name*, so `strategy.name` always
    matches what the orchestrator writes to the trade log.

    Parameters
    ----------
    regime_map:
        ``{regime: [strategy, ...]}``.  Candidates are in priority order; the
        first one wins unless `select` is overridden.  Entries may be registry
        names, Strategy subclasses or Strategy instances.  Defaults to the
        registry's `REGIME_STRATEGY_MAP`.
    fallback:
        Strategy to use when the regime is unknown, missing, or has no
        candidates.  ``None`` (default) means route to cash.
    """

    def __init__(
        self,
        regime_map: Optional[Mapping[str, Sequence[StrategySpec]]] = None,
        fallback: Optional[StrategySpec] = None,
    ):
        source = REGIME_STRATEGY_MAP if regime_map is None else regime_map
        self._instances: Dict[str, Strategy] = {}
        self._explicit: set[str] = set()  # names whose instance the caller supplied
        self._map: Dict[str, List[str]] = {}
        for regime, specs in source.items():
            key = self.normalise_regime(regime)
            if key is None:
                raise ValueError(f"Invalid regime key in regime_map: {regime!r}")
            self._map[key] = [self._register(spec) for spec in specs]
        self._fallback_name: Optional[str] = self._register(fallback) if fallback is not None else None

    # ------------------------------------------------------------------ config
    @property
    def regime_map(self) -> Dict[str, Tuple[str, ...]]:
        """Read-only view of the effective routing table (regime -> strategy names)."""
        return {regime: tuple(names) for regime, names in self._map.items()}

    @property
    def known_regimes(self) -> Tuple[str, ...]:
        return tuple(self._map.keys())

    @property
    def fallback_name(self) -> Optional[str]:
        return self._fallback_name

    def candidates_for(self, regime: Any) -> List[str]:
        """Strategy names eligible for `regime` in priority order ([] if none)."""
        key = self.normalise_regime(regime)
        if key is None:
            return []
        return list(self._map.get(key, []))

    # ----------------------------------------------------------------- routing
    def route(self, regime_info: Union[Mapping[str, Any], str, None]) -> RoutingDecision:
        """
        Decide which strategy (if any) should run for the given regime.

        Accepts the dict returned by `RegimeDetector.latest_regime()`, a bare
        regime string, or ``None`` (detector failed / no data).  Never raises
        on bad input: anything unrecognised routes to the fallback.
        """
        raw = regime_info.get("regime") if isinstance(regime_info, Mapping) else regime_info
        regime = self.normalise_regime(raw)

        if regime is None:
            return self._fallback_decision(None, f"no usable regime label ({raw!r})")

        if regime not in self._map:
            return self._fallback_decision(regime, "regime not in routing table")

        candidates = tuple(self._map[regime])
        if not candidates:
            return self._fallback_decision(regime, "regime has no eligible strategies", candidates)

        chosen = self.select(regime, candidates)
        if chosen is None:
            return self._fallback_decision(regime, "selector declined all candidates", candidates)
        if chosen not in candidates:
            raise ValueError(f"select() returned {chosen!r} which is not a candidate for {regime!r}")

        return RoutingDecision(
            regime=regime,
            strategy=self.get_strategy(chosen),
            reason="priority" if len(candidates) > 1 else "only candidate",
            candidates=candidates,
        )

    def select(self, regime: str, candidates: Sequence[str]) -> Optional[str]:
        """
        Pick one strategy name from `candidates` for `regime`.

        Default: highest priority (first).  Override to rank by live health
        metrics, recent performance, etc.  Return ``None`` to decline all
        candidates and fall back.
        """
        return candidates[0] if candidates else None

    # --------------------------------------------------------------- instances
    def get_strategy(self, name: str) -> Strategy:
        """Cached strategy instance for a registry name (KeyError if unknown)."""
        if name not in self._instances:
            self._instances[name] = get_strategy_class(name)()
        return self._instances[name]

    def has_strategy(self, name: str) -> bool:
        """True if `name` can be served by this router without touching the registry."""
        return name in self._instances

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def normalise_regime(value: Any) -> Optional[str]:
        """Canonical regime key ('trend ' -> 'TREND'); None for missing/NaN/empty."""
        return normalise_regime(value)

    def _register(self, spec: StrategySpec) -> str:
        """Resolve a map entry to a registry-style name, caching instances eagerly
        for classes/instances so `get_strategy` never has to hit the registry for them."""
        if isinstance(spec, Strategy):
            # An explicitly supplied instance always wins over one we built
            # from a class/name; two *different* explicit instances sharing a
            # name would be ambiguous (the name is the trade-log identity).
            existing = self._instances.get(spec.name)
            if existing is not None and existing is not spec and spec.name in self._explicit:
                raise ValueError(f"Two different strategy instances supplied for name {spec.name!r}")
            self._instances[spec.name] = spec
            self._explicit.add(spec.name)
            return spec.name
        if isinstance(spec, type) and issubclass(spec, Strategy):
            instance = spec()
            self._instances.setdefault(instance.name, instance)
            return instance.name
        if isinstance(spec, str):
            get_strategy_class(spec)  # fail fast on typos at construction time
            return spec
        raise TypeError(f"Unsupported strategy spec {spec!r}; expected name, class or instance")

    def _fallback_decision(self, regime: Optional[str], why: str,
                           candidates: Iterable[str] = ()) -> RoutingDecision:
        if self._fallback_name is None:
            return RoutingDecision(regime, None, f"{why}; routing to cash",
                                   tuple(candidates), fallback_used=True)
        return RoutingDecision(regime, self.get_strategy(self._fallback_name),
                               f"{why}; using fallback {self._fallback_name}",
                               tuple(candidates), fallback_used=True)


__all__ = ["StrategyRouter", "RoutingDecision", "StrategySpec"]