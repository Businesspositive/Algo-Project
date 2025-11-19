# Project Setup

## 1. Python Environment

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## 2. Environment Variables

1. Copy `config/env.example` to `.env` in the repo root.
2. Fill in:
   - `client_id` / `client_secret`: from Fyers app dashboard.
   - `FYERS_REDIRECT_URI`: must match the redirect configured in the app.
   - `FYERS_AUTH_CODE`: optional; only needed once to seed the local token database.
3. Keep `.env` private (already ignored via `.gitignore`).

## 3. Token Initialization

Run `python main_regime_scan.py` (or any script that triggers `FyersDataSource`) once after setting the `.env`. The first run will exchange `FYERS_AUTH_CODE`, store tokens in `utils/fyers_tokens.db`, and subsequent runs will auto-refresh.

If the auth code expires before first use, obtain a fresh code from the Fyers login flow and update `.env`.

## 4. Running Components

- Regime scan: `python main_regime_scan.py`
- Paper orchestrator demo: `python orchestrator.py`

Both reuse the same data sources, regime detector, and strategy registry, so new strategies become available everywhere automatically.


