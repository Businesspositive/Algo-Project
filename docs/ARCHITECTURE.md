# Algo Trading Architecture

## Packages

- `core/data`: shared data-source interfaces.
  - `base.DataSource`: contract for fetchers.
  - `fyers.FyersDataSource`: authenticated NSE data via Fyers.
  - `yfinance.YFinanceDataSource`: fallback/historical source.
- Config/environment values live in `.env` (copy from `config/env.example`) and are loaded automatically where needed.
- `core/strategies`: plug-in strategies derived from `Strategy`.
  - `registry.py`: maps names ↔ classes and regimes ↔ strategies.
  - Implementations live in separate modules (RSI mean reversion, Donchian breakout, Momentum, etc.).
- `regimes/detector.py`: computes ATR/ADX/volatility features and assigns the latest regime label.
- `screens/daily_selector.py`: orchestrates universe scans by pulling data through a `DataSource`, detecting regimes, and querying the strategy registry for compatible models.
- `orchestrator.py`: live/paper execution loop that reuses the same strategies and data sources, enforces health gates, and routes orders through brokers.

## Adding a Strategy

1. Create a new module under `core/strategies/` that subclasses `Strategy`.
2. Implement `generate_signals` returning a DataFrame with at least a `signal` column (and optionally a `score` diagnostic).
3. Register the class in `core/strategies/registry.STRATEGY_REGISTRY` and reference the strategy name inside `REGIME_STRATEGY_MAP` for the regimes it should run in.
4. The strategy automatically becomes available to `DailySelector` and the orchestrator without touching other modules.


