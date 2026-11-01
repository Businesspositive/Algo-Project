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
# Conservative per-side transaction cost assumed when sizing, in percent
# (same convention as brokers.fees_pct: 0.1 == 0.1%).  Deliberately above the
# paper broker's 0.05% default so notional + fee always fits in cash.
ESTIMATED_FEE_PCT = 0.1

REGIME_MAP = {
    "Donchian_Breakout": {"allowed": ["VOL_BREAKOUT", "TREND", "NORMAL"]},
    "RSI_MeanReversion": {"allowed": ["RANGE"]},
    "Momentum": {"allowed": ["TREND", "NORMAL"]},
}
# ------------------------------------------------

# lightweight trade history persistence (CSV)
TRADE_LOG_CSV = "paper_trade_log.csv"
TRADE_LOG_COLUMNS = ['timestamp', 'strategy', 'symbol', 'side', 'qty', 'price',
                     'entry_price', 'fees', 'pnl']

def load_trade_history() -> pd.DataFrame:
    try:
        df = pd.read_csv(TRADE_LOG_CSV, parse_dates=['timestamp'])
    except FileNotFoundError:
        return pd.DataFrame(columns=TRADE_LOG_COLUMNS)
    # tolerate logs written before entry_price/fees were tracked
    for col in TRADE_LOG_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df

def append_trade_to_history(rec: Dict[str, Any]):
    df = load_trade_history()
    new_row = pd.DataFrame([rec]).reindex(columns=TRADE_LOG_COLUMNS)
    df = new_row if df.empty else pd.concat([df, new_row], ignore_index=True)
    df.to_csv(TRADE_LOG_CSV, index=False)

def find_open_entry(strategy_name: str, symbol: str) -> Dict[str, Any] | None:
    """
    Return the most recent BUY record for (strategy, symbol) that has not yet
    been closed by a later SELL, or None if the log shows no open round trip.
    """
    df = load_trade_history()
    df = df[(df['strategy'] == strategy_name) & (df['symbol'] == symbol)]
    if df.empty:
        return None
    df = df.sort_values('timestamp', kind='stable').reset_index(drop=True)
    sides = df['side'].astype(str).str.upper()
    buy_idx = df.index[sides == 'BUY']
    if len(buy_idx) == 0:
        return None
    last_buy = df.loc[buy_idx[-1]]
    sold_since = df.loc[(df.index > buy_idx[-1]) & (sides == 'SELL'), 'qty'].astype(float).sum()
    if sold_since + 1e-9 >= float(last_buy['qty']):
        return None  # fully closed (possibly across several partial exits)
    return last_buy.to_dict()

def compute_realised_pnl(entry_price: float, exit_price: float, qty: float, fees: float = 0.0) -> float:
    """
    Realised P&L for a closed long position, net of round-trip fees.
    Gross P&L can be recovered as pnl + fees.
    """
    return float((float(exit_price) - float(entry_price)) * float(qty) - float(fees))

# ---------- order execution handling ----------
# Broker status vocab. FyersPaperBroker returns 'filled'/'rejected';
# BinanceTestnetBroker passes through exchange statuses (NEW, PARTIALLY_FILLED,
# FILLED, CANCELED, EXPIRED, REJECTED) or its own 'rejected'/'error'.
FILLED_STATUSES = {"filled"}
PARTIAL_STATUSES = {"partially_filled", "partial"}
FAILED_STATUSES = {"rejected", "error", "canceled", "cancelled", "expired", "new", "pending"}


class OrderOutcome:
    """Normalised view of a broker's place_order() response."""

    def __init__(self, status: str, filled_qty: float = 0.0, fill_price: float | None = None,
                 fee: float = 0.0, reason: str | None = None, raw: Any = None):
        self.status = status
        self.filled_qty = float(filled_qty)
        self.fill_price = fill_price
        self.fee = float(fee)
        self.reason = reason
        self.raw = raw

    @property
    def filled(self) -> bool:
        return self.status in ("filled", "partially_filled") and self.filled_qty > 0

    @property
    def partial(self) -> bool:
        return self.status == "partially_filled"

    def __repr__(self) -> str:
        return (f"OrderOutcome(status={self.status!r}, filled_qty={self.filled_qty}, "
                f"fill_price={self.fill_price}, fee={self.fee}, reason={self.reason!r})")


def interpret_order_result(res: Any, requested_qty: float) -> OrderOutcome:
    """
    Classify a broker response into filled / partially_filled / <failure>.
    Never raises: malformed responses are treated as failures so the cycle
    can log and move on without recording a trade that did not happen.
    """
    if not isinstance(res, dict):
        return OrderOutcome("error", reason=f"malformed broker response: {res!r}", raw=res)

    status = str(res.get("status") or "").strip().lower()
    reason = res.get("reason") or res.get("error")
    filled_qty = _to_float(res.get("filled_qty"), default=0.0)
    fill_price = _to_float(res.get("avg_fill_price"), default=None)
    fee = _to_float(res.get("fee"), default=0.0)

    if status in FILLED_STATUSES:
        if filled_qty <= 0:
            filled_qty = float(requested_qty)
        if fill_price is None or fill_price <= 0:
            return OrderOutcome("error", reason="filled without a fill price", raw=res)
        if filled_qty + 1e-9 < float(requested_qty):
            return OrderOutcome("partially_filled", filled_qty, fill_price, fee, reason, res)
        return OrderOutcome("filled", filled_qty, fill_price, fee, reason, res)

    if status in PARTIAL_STATUSES:
        if filled_qty <= 0 or fill_price is None or fill_price <= 0:
            return OrderOutcome("error", reason="partial fill without qty/price", raw=res)
        return OrderOutcome("partially_filled", filled_qty, fill_price, fee, reason, res)

    if status in FAILED_STATUSES:
        return OrderOutcome(status, 0.0, None, 0.0, reason, res)

    return OrderOutcome("error", reason=f"unknown broker status {status!r}", raw=res)


def submit_order(broker, symbol: str, side: str, qty: float) -> OrderOutcome:
    """Place an order and normalise the result; broker exceptions become an 'error' outcome."""
    try:
        res = broker.place_order(symbol, side, qty)
    except Exception as exc:  # broker/network failure must not kill the cycle
        print(f"Broker raised while placing {side} {qty} {symbol}: {exc!r}")
        return OrderOutcome("error", reason=f"exception: {exc!r}")
    outcome = interpret_order_result(res, qty)
    print("Order result:", res)
    if not outcome.filled:
        print(f"{side} order for {symbol} NOT executed -> status={outcome.status!r}, reason={outcome.reason!r}")
    elif outcome.partial:
        print(f"{side} order for {symbol} PARTIALLY filled: {outcome.filled_qty}/{qty}")
    return outcome


def _to_float(value: Any, default):
    try:
        if value is None:
            return default
        f = float(value)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default

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

def compute_position_qty(portfolio_value: float, price: float,
                         exposure_frac: float = MAX_EXPOSURE_PER_TRADE,
                         cash_available: float | None = None,
                         fee_pct: float = ESTIMATED_FEE_PCT) -> int:
    """
    Whole-share quantity for a new position sized as `exposure_frac` of total
    portfolio value (cash + open positions at market).

    Sizing off portfolio value -- not spare cash -- keeps the per-trade limit on
    the same denominator as MAX_TOTAL_EXPOSURE, so every entry consumes a
    predictable 5% slice of the 25% budget regardless of how much is already
    deployed.

    The order is additionally capped so that *notional + estimated fee* fits in
    `cash_available`: the broker charges fee_pct% of notional on top of the
    notional, so the most we can spend on shares is cash / (1 + fee_pct/100).
    Capping on notional alone would let an order sized exactly to cash be
    rejected for insufficient funds once the fee is added.
    """
    if price is None or price <= 0 or portfolio_value is None or portfolio_value <= 0:
        return 0
    invest_amount = float(portfolio_value) * float(exposure_frac)
    if cash_available is not None:
        fee_rate = max(0.0, float(fee_pct or 0.0)) / 100.0
        spendable = max(0.0, float(cash_available)) / (1.0 + fee_rate)
        invest_amount = min(invest_amount, spendable)
    qty = math.floor(invest_amount / float(price))
    return max(0, int(qty))


def estimated_fee_pct_for(broker) -> float:
    """
    Fee rate (percent) to assume when sizing: never less than ESTIMATED_FEE_PCT,
    and never less than what the broker says it actually charges.
    """
    try:
        actual = float(getattr(broker, 'fees_pct', 0.0) or 0.0)
    except (TypeError, ValueError):
        actual = 0.0
    return max(ESTIMATED_FEE_PCT, actual)


# ---------- portfolio-level risk ----------
def compute_portfolio_exposure(broker, price_lookup: Dict[str, float] | None = None) -> Dict[str, float]:
    """
    Snapshot of how much of the portfolio is currently deployed across *all*
    open positions, valued at market where a price is available.

    Returns cash, positions_value, portfolio_value and exposure
    (= positions_value / portfolio_value, 0..1).
    """
    cash = float(broker.get_balance()['cash'])
    positions = broker.get_positions() or {}
    if not positions:
        return {"cash": cash, "positions_value": 0.0, "portfolio_value": cash, "exposure": 0.0}

    try:
        portfolio_value = float(broker.portfolio_value(price_lookup))
        positions_value = portfolio_value - cash
    except Exception as exc:  # a flaky quote must not disable risk checks
        print(f"portfolio_value() failed ({exc!r}); valuing open positions at cost basis")
        positions_value = 0.0
        for sym, pos in positions.items():
            px = (price_lookup or {}).get(sym) or pos.get('avg_price') or 0.0
            positions_value += float(pos.get('qty', 0.0)) * float(px)
        portfolio_value = cash + positions_value

    positions_value = max(0.0, positions_value)
    exposure = positions_value / portfolio_value if portfolio_value > 0 else float('inf')
    return {
        "cash": cash,
        "positions_value": positions_value,
        "portfolio_value": portfolio_value,
        "exposure": exposure,
    }


def check_total_exposure(broker, symbol: str, qty: float, price: float,
                         max_total_exposure: float = MAX_TOTAL_EXPOSURE,
                         price_lookup: Dict[str, float] | None = None,
                         snapshot: Dict[str, float] | None = None) -> tuple[bool, Dict[str, float]]:
    """
    Would adding `qty` of `symbol` at `price` keep total portfolio exposure
    within `max_total_exposure`?  Returns (allowed, details).

    Buying converts cash into stock, so the portfolio value is unchanged by
    the trade (fees aside) and the projected exposure is simply
    (current positions value + new notional) / portfolio value.
    """
    if snapshot is None:
        lookup = dict(price_lookup or {})
        lookup.setdefault(symbol, price)  # value any existing holding of this symbol at the signal price
        snapshot = compute_portfolio_exposure(broker, lookup)
    snap = snapshot
    new_notional = float(qty) * float(price)
    pv = snap["portfolio_value"]
    projected = (snap["positions_value"] + new_notional) / pv if pv > 0 else float('inf')
    allowed = projected <= max_total_exposure + 1e-9
    details = {
        **snap,
        "new_notional": new_notional,
        "projected_exposure": projected,
        "max_total_exposure": max_total_exposure,
    }
    return allowed, details

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

    price = float(latest['Close'])
    position = broker.get_positions().get(symbol)
    in_position = bool(position) and float(position.get('qty', 0)) > 0

    # ---------------- EXIT WORKFLOW ----------------
    # A flat signal while we hold the symbol closes the round trip and realises P&L.
    # Exits deliberately bypass the health/regime gates: we must always be able
    # to flatten an open position regardless of the strategy's recent stats.
    if in_position:
        if signal == 1:
            print("Already in position and signal still long. Holding.")
            return
        _exit_position(strategy, symbol, broker, position, price)
        return

    # ---------------- ENTRY WORKFLOW ----------------
    if signal == 0:
        print("No trade signal and no open position. Done.")
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

    # 5) risk sizing: MAX_EXPOSURE_PER_TRADE of *total portfolio value*
    # (cash + open positions at market), capped so notional + fees fit in cash.
    snapshot = compute_portfolio_exposure(broker, {symbol: price})
    fee_pct = estimated_fee_pct_for(broker)
    qty = compute_position_qty(snapshot['portfolio_value'], price,
                               exposure_frac=MAX_EXPOSURE_PER_TRADE,
                               cash_available=snapshot['cash'], fee_pct=fee_pct)
    if qty <= 0:
        print(f"Sizing produced 0 shares (portfolio {snapshot['portfolio_value']:.2f}, "
              f"cash {snapshot['cash']:.2f}, price {price}). Skipping.")
        return
    print(f"Sized {qty} shares = {qty * price:.2f} "
          f"({qty * price / snapshot['portfolio_value']:.2%} of portfolio, target {MAX_EXPOSURE_PER_TRADE:.0%})")

    # 6) portfolio exposure gate: per-trade sizing is not enough on its own --
    # the sum of every open position plus this order must stay within
    # MAX_TOTAL_EXPOSURE of portfolio value, otherwise the signal is skipped.
    allowed, exp = check_total_exposure(broker, symbol, qty, price,
                                        max_total_exposure=MAX_TOTAL_EXPOSURE, snapshot=snapshot)
    print(f"Exposure: current {exp['exposure']:.1%} "
          f"(positions {exp['positions_value']:.2f} / portfolio {exp['portfolio_value']:.2f}), "
          f"projected with this trade {exp['projected_exposure']:.1%}, limit {exp['max_total_exposure']:.0%}")
    if not allowed:
        print(f"Total exposure gate failed: {exp['projected_exposure']:.1%} would exceed "
              f"{exp['max_total_exposure']:.0%}. Skipping trade.")
        return

    # 7) place order
    _enter_position(strategy, symbol, broker, qty, price)


def _enter_position(strategy: Strategy, symbol: str, broker: FyersPaperBroker, qty: float, ref_price: float):
    """Open a long position and log it only if the broker actually filled it.
    pnl stays unrealised (NaN) until the exit leg."""
    print(f"Placing BUY of {qty} @ ~{ref_price}")
    outcome = submit_order(broker, symbol, 'BUY', qty)
    if not outcome.filled:
        print("Entry not executed; nothing logged.")
        return None

    fill_price = outcome.fill_price
    filled_qty = outcome.filled_qty
    fee = outcome.fee
    rec = {
        'timestamp': pd.Timestamp.now(),
        'strategy': strategy.name,
        'symbol': symbol,
        'side': 'BUY',
        'qty': filled_qty,
        'price': fill_price,
        'entry_price': fill_price,
        'fees': fee,
        'pnl': np.nan,  # open trade: realised on exit
    }
    append_trade_to_history(rec)
    print("Entry logged (paper).")
    return rec


def _exit_position(strategy: Strategy, symbol: str, broker: FyersPaperBroker,
                   position: Dict[str, Any], ref_price: float):
    """
    Close the full open position, compute realised P&L from entry vs exit price
    (net of entry + exit fees) and log it so it feeds the strategy's metrics.
    """
    qty = float(position['qty'])
    # Entry cost basis: the broker's average fill price is authoritative.
    entry_price = position.get('avg_price')
    open_entry = find_open_entry(strategy.name, symbol)
    if entry_price is None and open_entry is not None:
        entry_price = open_entry.get('entry_price', open_entry.get('price'))
    if entry_price is None or pd.isna(entry_price):
        print("Cannot determine entry price for open position; skipping exit.")
        return None
    entry_price = float(entry_price)

    entry_fee, entry_qty = 0.0, qty
    if open_entry is not None and pd.notna(open_entry.get('fees')):
        entry_fee = float(open_entry['fees'])
        entry_qty = _to_float(open_entry.get('qty'), default=qty) or qty

    print(f"Placing SELL of {qty} @ ~{ref_price} (entry {entry_price})")
    outcome = submit_order(broker, symbol, 'SELL', qty)
    if not outcome.filled:
        print("Exit not executed; position remains open.")
        return None

    exit_price = outcome.fill_price
    filled_qty = outcome.filled_qty
    exit_fee = outcome.fee
    # On a partial exit only the closed fraction of the entry fee is realised;
    # the remainder is attributed when the rest of the position is sold.
    realised_entry_fee = entry_fee * (filled_qty / entry_qty) if entry_qty > 0 else 0.0
    fees = realised_entry_fee + exit_fee
    pnl = compute_realised_pnl(entry_price, exit_price, filled_qty, fees)
    if outcome.partial:
        print(f"Partial exit: {filled_qty}/{qty} closed; {qty - filled_qty} remains open.")

    rec = {
        'timestamp': pd.Timestamp.now(),
        'strategy': strategy.name,
        'symbol': symbol,
        'side': 'SELL',
        'qty': filled_qty,
        'price': exit_price,
        'entry_price': entry_price,
        'fees': fees,
        'pnl': pnl,
    }
    append_trade_to_history(rec)
    print(f"Exit logged (paper). Realised P&L: {pnl:.2f} (fees {fees:.2f})")
    return rec

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