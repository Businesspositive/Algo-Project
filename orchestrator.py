from __future__ import annotations
import time
import math
import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime, timedelta
from core.data import DataSource, YFinanceDataSource
from core.strategies.base import Strategy
from core.strategies.registry import get_strategy_class
from regimes.detector import RegimeDetector
from brokers import FyersPaperBroker, BinanceTestnetBroker, default_price_provider

# For demo I'll create minimal placeholders if you need to swap
# NOTE: replace these placeholders with your actual classes
# class BreakoutStrategy:
#     def __init__(self, lookback_high=20, lookback_low=10):
#         self.name = "Breakout"
#         self.lb_h = lookback_high
#         self.lb_l = lookback_low
#     def generate_signals(self, df):
#         out = df.copy()
#         out['DC_H'] = out['High'].rolling(self.lb_h).max().shift(1)
#         out['signal'] = (out['Close'] > out['DC_H']).astype(int)
#         return out

# class RegimeDetector:
#     def __init__(self, thresholds=None): pass
#     def latest_regime(self, df):
#         # crude placeholder: choose TREND if 20-day slope positive
#         ma = df['Close'].rolling(20).mean()
#         slope = (ma.iloc[-1] - ma.iloc[-10]) if len(ma)>10 else 0
#         regime = "TREND" if slope>0 else "RANGE"
#         return {"regime": regime, "adx":0.0, "bb_width":0.0, "rvol":0.0, "atr_frac":0.0}

# -------------- Config / thresholds -------------
SHARPE_MIN = 1.2
PF_MIN = 1.2
MIN_TRADES_FOR_STAT = 10
MAXDD_THRESHOLD = -0.30
MAX_EXPOSURE_PER_TRADE = 0.05
MAX_TOTAL_EXPOSURE = 0.25

REGIME_MAP = {
    "Donchian_Breakout": {"allowed": ["VOL_BREAKOUT", "TREND", "NORMAL"]},
    "RSI_MeanReversion": {"allowed": ["RANGE"]},
    "Momentum": {"allowed": ["TREND", "NORMAL"]},
}
# ------------------------------------------------

# lightweight trade history persistence (CSV)
TRADE_LOG_CSV = "paper_trade_log.csv"

def load_trade_history() -> pd.DataFrame:
    try:
        return pd.read_csv(TRADE_LOG_CSV, parse_dates=['timestamp'])
    except FileNotFoundError:
        return pd.DataFrame(columns=['timestamp','strategy','symbol','side','qty','price','pnl'])

def append_trade_to_history(rec: Dict[str, Any]):
    df = load_trade_history()
    df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)
    df.to_csv(TRADE_LOG_CSV, index=False)

def recent_metrics_from_history(strategy_name: str, window_days: int = 90, min_trades: int = MIN_TRADES_FOR_STAT):
    df = load_trade_history()
    df = df[df['strategy']==strategy_name].dropna(subset=['pnl'])
    if df.empty or len(df) < min_trades:
        return {"n_trades": len(df), "sharpe": None, "pf": None, "maxdd": None}
    # take only recent window
    window_start = pd.Timestamp.now() - pd.Timedelta(days=window_days)
    dfw = df[df['timestamp'] >= window_start]
    if dfw.empty or len(dfw) < min_trades:
        dfw = df.tail(min_trades)  # fallback
    returns = dfw['pnl']
    # simple stats
    wins = returns[returns>0].sum()
    losses = -returns[returns<0].sum()
    pf = (wins / losses) if losses>0 else float('inf')
    # daily equity series approximation (cumulative pnl)
    eq = (returns.cumsum() + 1.0)  # synthetic equity curve starting at 1
    daily_rets = eq.pct_change().fillna(0)
    sharpe = (daily_rets.mean() / (daily_rets.std()+1e-12)) * (252**0.5)
    # maxdd
    roll_max = eq.cummax()
    dd = eq/roll_max - 1.0
    maxdd = dd.min()
    return {"n_trades": len(dfw), "sharpe": float(sharpe), "pf": float(pf), "maxdd": float(maxdd)}

def compute_position_qty(cash_balance: float, price: float, exposure_frac: float = MAX_EXPOSURE_PER_TRADE):
    # invest exposure_frac fraction of cash_balance
    invest_amount = cash_balance * exposure_frac
    qty = math.floor(invest_amount / price)
    return max(0, qty)

# ---------- Orchestrator / main loop ----------
def orchestrate_one_cycle(
    strategy: Strategy,
    symbol: str,
    broker: FyersPaperBroker,
    data_source: DataSource | None = None,
    lookback_days: int = 180,
):
    print(f"[{datetime.now()}] Orchestrator: symbol={symbol}, strategy={strategy.name}")
    data_source = data_source or YFinanceDataSource()
    end_date = datetime.today()
    start_date = end_date - timedelta(days=lookback_days)
    df = data_source.fetch(
        symbol,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        interval="1d",
    )
    if df is None or df.empty:
        print("No data, skipping")
        return

    # 1) detect regime
    detector = RegimeDetector()
    regime_info = detector.latest_regime(df)
    regime = regime_info['regime']
    print("Regime:", regime)

    # 2) generate signals
    sig_df = strategy.generate_signals(df)
    latest = sig_df.iloc[-1]
    signal = int(latest.get('signal', 0))
    print("Latest signal:", signal)

    if signal == 0:
        print("No trade signal. Done.")
        return

    # 3) strategy health gate
    metrics = recent_metrics_from_history(strategy.name)
    print("Recent metrics:", metrics)
    if (metrics.get('sharpe') is None) or (metrics['sharpe'] < SHARPE_MIN):
        print("Strategy health gate failed (Sharpe). Skipping trade.")
        return
    if (metrics.get('pf') is None) or (metrics['pf'] < PF_MIN):
        print("Strategy health gate failed (PF). Skipping trade.")
        return
    if (metrics.get('maxdd') is not None) and (metrics['maxdd'] < MAXDD_THRESHOLD):
        print("Strategy drawdown too deep. Skipping.")
        return

    # 4) regime gate
    allowed = REGIME_MAP.get(strategy.name, {}).get('allowed', [])
    if allowed and (regime not in allowed):
        print(f"Regime gate failed: {regime} not in {allowed}")
        return

    # 5) risk sizing & place order
    price = float(latest['Close'])
    cash = broker.get_balance()['cash']
    qty = compute_position_qty(cash, price)
    if qty <= 0:
        print("Not enough cash to buy any shares.")
        return

    side = 'BUY' if signal==1 else 'SELL'
    print(f"Placing {side} of {qty} @ {price}")
    res = broker.place_order(symbol, side, qty)
    print("Order result:", res)

    # log trade (we estimate pnl as NA until exit; for BUY we log price and qty)
    rec = {
        'timestamp': pd.Timestamp.now(),
        'strategy': strategy.name,
        'symbol': symbol,
        'side': side,
        'qty': qty,
        'price': price,
        'pnl': None
    }
    append_trade_to_history(rec)
    print("Trade logged (paper).")

# ---------- sample run ----------
if __name__ == "__main__":
    broker = FyersPaperBroker(
        initial_cash=200000.0,
        market_price_provider=default_price_provider,
        fees_pct=0.05,
    )
    strat_cls = get_strategy_class("Donchian_Breakout")
    strat = strat_cls()
    orchestrate_one_cycle(strat, "RELIANCE.NS", broker)
