import pytest

from core.strategies.donchian_breakout import DonchianBreakout
from core.strategies.momentum import MomentumStrategy
from core.strategies.rsi_mean_reversion import RSIMeanReversion


@pytest.mark.parametrize(
    "strategy_class",
    [
        DonchianBreakout,
        MomentumStrategy,
        RSIMeanReversion,
    ],
)
def test_strategy_returns_signal_column(strategy_class, sample_ohlcv):
    strategy = strategy_class()

    result = strategy.generate_signals(sample_ohlcv)

    assert not result.empty
    assert "signal" in result.columns
    assert result["signal"].notna().all()
    assert result["signal"].isin([-1, 0, 1]).all()