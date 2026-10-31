# Development Guide

## Project Workflows

Algo-Project contains two primary workflows.

### 1. Market Screening

The screening workflow:

1. Fetches historical market data.
2. Detects the current market regime.
3. Identifies strategies compatible with that regime.
4. Generates signals for selected instruments.
5. Returns suitable trading candidates.

Main components:

- `core/data/`
- `core/strategies/`
- `regimes/`
- `screens/`
- `main_regime_scan.py`

### 2. Paper Trading

The paper-trading workflow:

1. Fetches historical market data.
2. Detects the current market regime.
3. Generates strategy signals.
4. Evaluates strategy performance.
5. Applies regime restrictions.
6. Calculates order quantity.
7. Sends orders to a simulated broker.
8. Records trade activity.

Main components:

- `orchestrator.py`
- `brokers.py`
- `core/data/`
- `core/strategies/`
- `regimes/`

## Adding a Strategy

1. Create a strategy class under `core/strategies/`.
2. Extend the `Strategy` base class.
3. Implement `generate_signals()`.
4. Register the strategy in `core/strategies/registry.py`.
5. Add tests for the strategy.
6. Verify compatibility with screening and paper trading.

## Testing Guidelines

- Use deterministic sample market data.
- Do not call external APIs from unit tests.
- Use simulated brokers for paper-trading tests.
- Test expected behaviour and error cases.
- Run the full test suite before submitting changes.

## Security Guidelines

- Never commit API keys, access tokens, PINs or passwords.
- Keep authentication information in environment variables.
- Do not print sensitive authentication payloads.
- Do not use live trading credentials in development or tests.