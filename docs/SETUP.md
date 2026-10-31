# Project Setup

## 1. Prerequisites

- Python 3.10 or later
- pip
- Git

## 2. Create a virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

## 3. Install dependencies

For application execution:

```bash
pip install -r requirements.txt
```

For development and testing:

```bash
pip install -r requirements-dev.txt
```

## 4. Configure Fyers credentials

Copy the example configuration:

```bash
cp config/env.example .env
```

Configure the following values when using Fyers:

- `client_id`
- `client_secret`
- `FYERS_REDIRECT_URI`
- `FYERS_AUTH_CODE` when generating the initial token
- `FYERS_PIN` when required for token refresh

Never commit `.env` or real authentication credentials.

## 5. Running the application

### Market screening

```bash
python main_regime_scan.py
```

This workflow currently uses Fyers market data and requires valid authentication.

### Paper trading

```bash
python orchestrator.py
```

The orchestrator currently uses its configured default data provider and simulated broker.

## 6. Running tests

```bash
pytest -q
```

Tests should use local fixtures and simulated components. They must not require live broker credentials or submit actual orders.

## 7. Important notes

- The paper broker is a simulator.
- BinanceTestnetBroker is a separate testnet integration.
- FyersDataSource requires Fyers authentication.
- External market-data providers may require internet access.
- Never use production broker credentials in automated tests.