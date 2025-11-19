from __future__ import annotations

from typing import Dict, List, Type

from .base import Strategy
from .donchian_breakout import DonchianBreakout
from .momentum import MomentumStrategy
from .rsi_mean_reversion import RSIMeanReversion

STRATEGY_REGISTRY: Dict[str, Type[Strategy]] = {
    "RSI_MeanReversion": RSIMeanReversion,
    "Donchian_Breakout": DonchianBreakout,
    "Momentum": MomentumStrategy,
}

REGIME_STRATEGY_MAP: Dict[str, List[str]] = {
    "RANGE": ["RSI_MeanReversion"],
    "VOL_BREAKOUT": ["Donchian_Breakout"],
    "TREND": ["Donchian_Breakout", "Momentum"],
    "NORMAL": ["Momentum"],
}


def get_strategy_class(name: str) -> Type[Strategy]:
    if name not in STRATEGY_REGISTRY:
        raise KeyError(f"Strategy {name} not registered.")
    return STRATEGY_REGISTRY[name]


def strategies_for_regime(regime: str) -> List[Strategy]:
    names = REGIME_STRATEGY_MAP.get(regime, [])
    return [get_strategy_class(name)() for name in names]


