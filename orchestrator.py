from __future__ import annotations
import time
import math
import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime, timedelta
from core.data import DataSource, YFinanceDataSource
from core.strategies.base import Strategy
from core.strategies.registry import REGIME_STRATEGY_MAP, get_strategy_class
from core.strategies.router import RoutingDecision, StrategyRouter
from regimes.detector import RegimeDetector
from regimes.stability import DEFAULT_MIN_STABLE_PERIODS, RegimeCooldown, RegimeStability
from brokers import FyersPaperBroker, BinanceTestnetBroker, default_price_provider

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

# Per-strategy regime gate, derived from the registry's regime -> strategies
# table so the orchestrator and the StrategyRouter can never disagree about
# where a strategy is allowed to trade.  Strategies absent from the registry
# (e.g. ad-hoc/test strategies) have no entry and are therefore unrestricted.
REGIME_MAP: Dict[str, Dict[str, list]] = {}
for _regime, _names in REGIME_STRATEGY_MAP.items():
    for _name in _names:
        REGIME_MAP.setdefault(_name, {"allowed": []})["allowed"].append(_regime)
del _regime, _names, _name
# ------------------------------------------------

# lightweight trade history persistence (CSV)
TRADE_LOG_CSV = "paper_trade_log.csv"
TRADE_LOG_COLUMNS = ['timestamp', 'strategy', 'symbol', 'side', 'qty', 'price',
                     'entry_price', 'fees', 'pnl', 'reason']

# Values written to the trade log's `reason` column.
REASON_SIGNAL = "signal"                 # the strategy's own signal opened/closed the trade
REASON_REGIME_CASH = "regime_cash"       # forced: regime routed to cash (unknown / QUIET / detector failure)
REASON_REGIME_SHIFT = "regime_shift"     # forced: owner no longer eligible in the new regime
REASON_UNATTRIBUTED = "unattributed"     # forced: open position with no owner in the trade log
REASON_MANUAL = "manual"                 # forced: explicit flatten_all_positions() call

# Strategy label used when a position has to be closed but the trade log
# cannot say which strategy opened it (e.g. log lost, or opened out-of-band).
UNATTRIBUTED_STRATEGY = "Unattributed"

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

def find_position_owner(symbol: str) -> str | None:
    """
    Name of the strategy whose BUY opened the currently open position in
    `symbol`, or None if the log shows no open round trip for it.

    Regimes (and therefore the routed strategy) change between cycles, but a
    position must be *closed* by the strategy that opened it so the SELL pairs
    with the right BUY and realised P&L is attributed to the right strategy.
    If several strategies somehow hold the symbol at once, the most recently
    opened one wins.
    """
    df = load_trade_history()
    df = df[df['symbol'] == symbol].dropna(subset=['strategy'])
    if df.empty:
        return None
    owner, owner_ts = None, None
    for name in df['strategy'].astype(str).unique():
        entry = find_open_entry(name, symbol)
        if entry is None:
            continue
        ts = pd.Timestamp(entry['timestamp'])
        if owner_ts is None or ts > owner_ts:
            owner, owner_ts = name, ts
    return owner

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


# ---------- unsettled liquidations ----------
class PendingLiquidation:
    """
    A sell whose proceeds we do not yet trust as spendable and whose shares we
    do not yet trust as off the book (fill not propagated, T+n settlement...).

    pre_qty  -- holding *before* the sell, so the exposure snapshot can tell
                which part of the sale the broker already reflects.
    qty      -- shares sold.
    price    -- average fill price.
    proceeds -- net cash expected from the sale (notional minus fees).
    """

    def __init__(self, symbol: str, qty: float, price: float, proceeds: float,
                 pre_qty: float, timestamp: pd.Timestamp | None = None, reason: str = ""):
        self.symbol = symbol
        self.qty = float(qty)
        self.price = float(price)
        self.proceeds = max(0.0, float(proceeds))
        self.pre_qty = float(pre_qty)
        self.timestamp = timestamp or pd.Timestamp.now()
        self.reason = reason

    @property
    def notional(self) -> float:
        return self.qty * self.price

    def unreflected_qty(self, shown_qty: float) -> float:
        """Shares of this sale that `get_positions()` no longer reports (0 if the
        broker still shows the full pre-sale holding, i.e. it is already counted)."""
        return max(0.0, min(self.qty, self.pre_qty - float(shown_qty or 0.0)))

    def __repr__(self) -> str:
        return (f"PendingLiquidation({self.symbol}, qty={self.qty}, price={self.price}, "
                f"proceeds={self.proceeds:.2f}, reason={self.reason!r})")


class SettlementLedger:
    """
    Liquidations whose proceeds are still unsettled.

    The orchestrator records every forced flatten here and the risk gates
    consult it, so freed capital cannot be redeployed -- and the liquidated
    exposure is not considered gone -- until settlement.  Share one ledger
    across a sweep of symbols so a flatten in one cycle is respected by the
    entries of every later cycle in the same run.

    settlement_delay:
        None  -> entries stay pending until `clear()` / `settle()` is called
                 (the right default for a ledger that lives for one sweep).
        timedelta -> entries expire automatically once that much time has
                 passed since the fill (e.g. timedelta(days=1) for T+1).
    """

    def __init__(self, settlement_delay: timedelta | None = None):
        self.settlement_delay = settlement_delay
        self._items: list[PendingLiquidation] = []

    def record(self, liq: PendingLiquidation | None) -> None:
        if liq is not None and liq.qty > 0:
            self._items.append(liq)

    def pending(self, now: pd.Timestamp | None = None) -> list[PendingLiquidation]:
        if self.settlement_delay is None:
            return list(self._items)
        now = now or pd.Timestamp.now()
        self._items = [x for x in self._items if x.timestamp + self.settlement_delay > now]
        return list(self._items)

    def settle(self, symbol: str | None = None) -> None:
        """Mark everything (or one symbol) as settled."""
        self._items = [] if symbol is None else [x for x in self._items if x.symbol != symbol]

    clear = settle

    @property
    def unsettled_proceeds(self) -> float:
        return sum(x.proceeds for x in self.pending())

    def __len__(self) -> int:
        return len(self.pending())

    def __repr__(self) -> str:
        return f"SettlementLedger(pending={self.pending()!r})"


def _pending_list(pending) -> list[PendingLiquidation]:
    if pending is None:
        return []
    if isinstance(pending, SettlementLedger):
        return pending.pending()
    if isinstance(pending, PendingLiquidation):
        return [pending]
    return list(pending)


# ---------- portfolio-level risk ----------
def compute_portfolio_exposure(broker, price_lookup: Dict[str, float] | None = None,
                               pending=None) -> Dict[str, float]:
    """
    Snapshot of how much of the portfolio is currently deployed across *all*
    open positions, valued at market where a price is available.

    `pending` (SettlementLedger / PendingLiquidation / iterable of them) adds
    the strict treatment of unsettled liquidations:

      in_flight_value     -- market value of sold shares the broker no longer
                             shows but whose sale is not yet settled.  It is
                             counted as exposure: until settled, that risk is
                             not considered off the book.
      unsettled_proceeds  -- cash from those sales we may not spend yet.
      settled_cash        -- cash minus unsettled_proceeds (floored at 0): the
                             only cash position sizing may use.

    Returns cash, settled_cash, positions_value, in_flight_value,
    unsettled_proceeds, portfolio_value and exposure
    (= (positions_value + in_flight_value) / portfolio_value, 0..1).
    Without `pending` the extra fields are zero and exposure is the plain
    positions_value / portfolio_value.
    """
    cash = float(broker.get_balance()['cash'])
    positions = broker.get_positions() or {}
    if not positions:
        positions_value, portfolio_value = 0.0, cash
    else:
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

    in_flight_value, unsettled_proceeds = 0.0, 0.0
    for liq in _pending_list(pending):
        shown_qty = float((positions.get(liq.symbol) or {}).get('qty', 0.0) or 0.0)
        px = (price_lookup or {}).get(liq.symbol) or liq.price
        in_flight_value += liq.unreflected_qty(shown_qty) * float(px)
        unsettled_proceeds += liq.proceeds

    deployed = positions_value + in_flight_value
    exposure = deployed / portfolio_value if portfolio_value > 0 else (float('inf') if deployed > 0 else 0.0)
    return {
        "cash": cash,
        "settled_cash": max(0.0, cash - unsettled_proceeds),
        "positions_value": positions_value,
        "in_flight_value": in_flight_value,
        "unsettled_proceeds": unsettled_proceeds,
        "portfolio_value": portfolio_value,
        "exposure": exposure,
    }


def check_total_exposure(broker, symbol: str, qty: float, price: float,
                         max_total_exposure: float = MAX_TOTAL_EXPOSURE,
                         price_lookup: Dict[str, float] | None = None,
                         snapshot: Dict[str, float] | None = None,
                         pending=None) -> tuple[bool, Dict[str, float]]:
    """
    Would adding `qty` of `symbol` at `price` keep total portfolio exposure
    within `max_total_exposure`?  Returns (allowed, details).

    Buying converts cash into stock, so the portfolio value is unchanged by
    the trade (fees aside) and the projected exposure is simply
    (open positions + unsettled liquidations + new notional) / portfolio value.
    """
    if snapshot is None:
        lookup = dict(price_lookup or {})
        lookup.setdefault(symbol, price)  # value any existing holding of this symbol at the signal price
        snapshot = compute_portfolio_exposure(broker, lookup, pending=pending)
    snap = snapshot
    new_notional = float(qty) * float(price)
    pv = snap["portfolio_value"]
    deployed = snap["positions_value"] + snap.get("in_flight_value", 0.0)
    projected = (deployed + new_notional) / pv if pv > 0 else float('inf')
    allowed = projected <= max_total_exposure + 1e-9
    details = {
        **snap,
        "new_notional": new_notional,
        "projected_exposure": projected,
        "max_total_exposure": max_total_exposure,
    }
    return allowed, details

# ---------- Orchestrator / main loop ----------
def detect_regime(df: pd.DataFrame, detector: RegimeDetector | None = None) -> Dict[str, Any] | None:
    """
    Latest regime snapshot for `df`, or None if the detector cannot produce one
    (too little history, NaN features, ...).  A failed detection must never
    abort the cycle: callers treat None as "unknown regime", which the router
    maps to cash.
    """
    detector = detector or RegimeDetector()
    try:
        return detector.latest_regime(df)
    except Exception as exc:
        print(f"Regime detection failed ({exc!r}); treating regime as unknown")
        return None


def regime_history(df: pd.DataFrame, detector: RegimeDetector | None = None) -> pd.Series | None:
    """
    Per-bar regime labels for `df` (see RegimeDetector.regime_history), or None
    if the detector cannot produce them.  None makes the cooldown treat the
    regime as unknown, which blocks new entries -- the safe direction.
    """
    detector = detector or RegimeDetector()
    try:
        return detector.regime_history(df)
    except Exception as exc:
        print(f"Regime history unavailable ({exc!r}); treating regime as unstable")
        return None


class _UnattributedOwner(Strategy):
    """
    Stand-in owner for a position the trade log cannot attribute to any
    strategy.  It always wants to be flat, so whatever path handles it ends
    in a market sell logged under UNATTRIBUTED_STRATEGY rather than polluting
    a real strategy's P&L history.
    """

    def __init__(self):
        super().__init__(UNATTRIBUTED_STRATEGY)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal"] = 0
        return out


def strategy_eligible_in_regime(strategy_name: str, regime: str | None,
                                router: StrategyRouter | None = None) -> bool:
    """
    May `strategy_name` hold risk in `regime`?

    Routed mode (router given): eligible iff the router lists it as a
    candidate for the regime -- an unknown/cash regime has no candidates, so
    nothing is eligible.  Pinned mode: the registry-derived REGIME_MAP gate;
    strategies absent from it are unrestricted.
    """
    if router is not None:
        return strategy_name in router.candidates_for(regime)
    allowed = REGIME_MAP.get(strategy_name, {}).get('allowed', [])
    return (not allowed) or (regime in allowed)


class CyclePlan:
    """
    What one cycle should do, decided before any signal is generated.

    flatten / flatten_reason:
        If set, the open position must first be force-closed with a market
        sell, logged under `flatten` (the owner, or the Unattributed stand-in).
    strategy:
        Strategy whose signal drives the rest of the cycle once flat (or the
        owner, when it simply keeps managing its position).  None = nothing
        further to do.
    """

    def __init__(self, strategy: Strategy | None, reason: str,
                 flatten: Strategy | None = None, flatten_reason: str | None = None):
        self.strategy = strategy
        self.reason = reason
        self.flatten = flatten
        self.flatten_reason = flatten_reason

    def __repr__(self) -> str:
        s = self.strategy.name if self.strategy else None
        f = self.flatten.name if self.flatten else None
        return f"CyclePlan(strategy={s!r}, flatten={f!r}, flatten_reason={self.flatten_reason!r}, reason={self.reason!r})"


def plan_cycle(
    decision: RoutingDecision,
    symbol: str,
    in_position: bool,
    router: StrategyRouter,
    pinned: Strategy | None = None,
    force_flatten: bool = True,
    stability: RegimeStability | None = None,
) -> CyclePlan:
    """
    Decide which strategy drives this cycle, and whether the open position
    must be force-closed first.

    Policy
    ------
    * Flat  -> the routed strategy (or the pinned one) may enter; nobody if
               the router chose cash.
    * Long, owner still eligible in the current regime -> the owner keeps
      managing the position (its own signal decides the exit), even if the
      router's first choice for the regime is a different strategy.  This
      keeps the SELL paired with the right BUY and P&L attributed correctly.
    * Long, owner NOT eligible (regime routed to cash, or a hard shift to a
      regime the owner does not trade) -> force-flatten under the owner's
      name, then hand the cycle to the newly routed strategy so fresh capital
      is only allocated once the old regime's risk is off the book.
    * Long, owner unknown -> force-flatten under UNATTRIBUTED_STRATEGY: a
      position nobody claims cannot be managed consistently.

    `force_flatten=False` restores the permissive behaviour: an ineligible or
    unknown owner keeps/adopts the position and exits on signal only.

    Regime cooldown (`stability`, see regimes.stability)
    ----------------------------------------------------
    When the current regime has not yet held for the required number of bars
    the regime is *in cooldown*: everything above that protects capital still
    happens immediately (an ineligible owner is flattened on the spot), but
    **no new position is opened** -- neither by the routed strategy nor by a
    strategy taking over after a flatten -- until the regime proves stable.
    The cooldown governs regime-aware strategies only: a pinned strategy that
    is absent from REGIME_MAP is unrestricted by regime and therefore also
    unaffected by the cooldown.
    """
    regime = decision.regime
    routed_next = None if pinned is not None else decision.strategy
    cooldown = (
        stability is not None and stability.in_cooldown
        and (pinned is None or pinned.name in REGIME_MAP)
    )
    if cooldown:
        routed_next = None
        cooldown_note = f"regime cooldown [{stability}]: no new positions"

    # ----- flat: who may enter --------------------------------------------
    if not in_position:
        if cooldown:
            return CyclePlan(None, cooldown_note)
        if pinned is not None:
            return CyclePlan(pinned, "pinned")
        if decision.is_cash:
            return CyclePlan(None, f"router chose cash ({decision.reason})")
        return CyclePlan(decision.strategy, f"routed ({decision.reason})")

    # ----- long: establish the owner ----------------------------------------
    if pinned is not None:
        owner: Strategy | None = pinned
        eligible = strategy_eligible_in_regime(pinned.name, regime)
    else:
        owner = None
        owner_name = find_position_owner(symbol)
        if owner_name is None:
            print(f"No trade-log owner found for open position in {symbol}")
        else:
            try:
                owner = router.get_strategy(owner_name)
            except KeyError:
                print(f"Position owner {owner_name!r} is not a registered strategy")
        if owner is None:
            if force_flatten:
                why = cooldown_note if cooldown else f"after flattening unattributed position ({decision.reason})"
                return CyclePlan(routed_next, why,
                                 flatten=_UnattributedOwner(), flatten_reason=REASON_UNATTRIBUTED)
            if decision.is_cash:
                return CyclePlan(None, "open position without owner and router chose cash; holding")
            return CyclePlan(decision.strategy, f"adopting unowned position ({decision.reason})")
        eligible = strategy_eligible_in_regime(owner.name, regime, router)

    # ----- long: keep or force out -------------------------------------------
    if eligible:
        return CyclePlan(owner, "owns the open position and is eligible in this regime")
    if not force_flatten:
        return CyclePlan(owner, f"owns the open position (not eligible in {regime!r}, but force_flatten=False)")

    if decision.is_cash:
        flatten_reason = f"{REASON_REGIME_CASH}:{regime}"
    else:
        flatten_reason = f"{REASON_REGIME_SHIFT}:{regime}"
    # Pinned mode: the pinned strategy is what just got flattened, so it may
    # not re-enter this cycle (the regime gate would refuse it anyway).
    if cooldown:
        next_reason = f"flattened {owner.name}; {cooldown_note}"
    elif routed_next is not None:
        next_reason = f"after flattening {owner.name} ({decision.reason})"
    else:
        next_reason = f"nothing to run after flattening {owner.name}"
    return CyclePlan(routed_next, next_reason, flatten=owner, flatten_reason=flatten_reason)


# ---------- forced flattening ----------
class FlattenResult:
    """Outcome of a forced flatten: was the position fully closed, and what
    (if anything) was sold and is now awaiting settlement."""

    def __init__(self, closed: bool, liquidation: PendingLiquidation | None):
        self.closed = closed
        self.liquidation = liquidation

    def __bool__(self) -> bool:  # so `if flatten_position(...)` still reads naturally
        return self.closed

    def __repr__(self) -> str:
        return f"FlattenResult(closed={self.closed}, liquidation={self.liquidation!r})"


def flatten_position(owner: Strategy, symbol: str, broker, position: Dict[str, Any],
                     ref_price: float, reason: str,
                     ledger: SettlementLedger | None = None) -> FlattenResult:
    """
    Market-sell the whole open position in `symbol`, logging the SELL under
    `owner` with the given reason.

    `closed` is True only if the position is now fully flat; a rejected or
    partial fill leaves it False so the caller does not put new capital to
    work on top of residual risk.  Whatever *was* sold is returned as a
    PendingLiquidation (and recorded in `ledger` if given) so the risk gates
    keep treating it as unsettled.
    """
    qty = float(position.get('qty', 0) or 0)
    print(f"FORCED FLATTEN ({reason}): closing {qty} {symbol} held by {owner.name}")
    rec = _exit_position(owner, symbol, broker, position, ref_price, reason=reason)
    if rec is None:
        print("Forced flatten NOT executed; position remains open.")
        return FlattenResult(False, None)

    sold_qty, fill_price = float(rec['qty']), float(rec['price'])
    # rec['fees'] also carries the prorated entry fee, so proceeds are slightly
    # *under*-stated -- the conservative direction for spendable cash.
    liq = PendingLiquidation(symbol, sold_qty, fill_price,
                             proceeds=sold_qty * fill_price - float(rec.get('fees') or 0.0),
                             pre_qty=qty, timestamp=rec['timestamp'], reason=reason)
    if ledger is not None:
        ledger.record(liq)

    remaining = qty - sold_qty
    if remaining > 1e-9:
        print(f"Forced flatten only partially filled; {remaining} {symbol} still open.")
        return FlattenResult(False, liq)
    return FlattenResult(True, liq)


def flatten_all_positions(broker, router: StrategyRouter | None = None,
                          price_lookup: Dict[str, float] | None = None,
                          reason: str = REASON_MANUAL,
                          ledger: SettlementLedger | None = None) -> Dict[str, FlattenResult]:
    """
    Kill switch: market-sell every open position, each logged under the
    strategy that opened it (or Unattributed).  Returns {symbol: FlattenResult};
    every sale is recorded in `ledger` when one is given.
    """
    router = router or StrategyRouter()
    results: Dict[str, FlattenResult] = {}
    for symbol, position in list((broker.get_positions() or {}).items()):
        if float(position.get('qty', 0) or 0) <= 0:
            continue
        owner: Strategy | None = None
        owner_name = find_position_owner(symbol)
        if owner_name is not None:
            try:
                owner = router.get_strategy(owner_name)
            except KeyError:
                pass
        owner = owner or _UnattributedOwner()
        ref_price = float((price_lookup or {}).get(symbol) or position.get('avg_price') or 0.0)
        results[symbol] = flatten_position(owner, symbol, broker, position, ref_price, reason, ledger=ledger)
    return results


def orchestrate_one_cycle(
    strategy: Strategy | None,
    symbol: str,
    broker: FyersPaperBroker,
    data_source: DataSource | None = None,
    lookback_days: int = 180,
    router: StrategyRouter | None = None,
    detector: RegimeDetector | None = None,
    force_flatten: bool = True,
    ledger: SettlementLedger | None = None,
    min_stable_periods: int = DEFAULT_MIN_STABLE_PERIODS,
    cooldown: RegimeCooldown | None = None,
):
    """
    Run one decision cycle for `symbol`.

    strategy:
        Pin a specific strategy for this cycle (it still has to pass the regime
        gate below).  Pass ``None`` to let `router` pick one from the detected
        regime -- the normal mode for live/paper trading.
    router:
        Regime -> strategy policy used when `strategy` is None.  Defaults to
        ``StrategyRouter()`` (the registry's REGIME_STRATEGY_MAP, cash fallback).
        Reuse one router across cycles so strategy instances are shared.
    force_flatten:
        Strict regime discipline (default).  If we hold `symbol` and the
        strategy that opened it is no longer eligible in the current regime
        (regime routed to cash, a hard shift to a regime it does not trade, or
        the position cannot be attributed at all), the position is closed with
        a market sell *before* the newly routed strategy may allocate capital.
        See `plan_cycle` for the full policy.
    ledger:
        SettlementLedger of liquidations whose proceeds are not yet settled.
        Any forced flatten in this cycle is recorded in it, and the sizing /
        total-exposure gates treat everything in it as still deployed and
        its cash as unspendable.  Defaults to a fresh ledger per cycle (i.e.
        strict treatment of this cycle's own liquidation); pass one shared
        ledger across a sweep of symbols so a flatten in one symbol's cycle
        is respected by entries in the next symbol's cycle.
    min_stable_periods:
        Regime-transition cooldown (default 3 bars).  New capital is only
        allocated once the detected regime has held for this many consecutive
        bars; until then ineligible positions are still flattened immediately
        but nothing new is opened, so a regime flickering on a threshold
        cannot churn sell/re-buy fees every day.  1 disables the cooldown.
        Labels in the same family (default: TREND + VOL_BREAKOUT) share one
        streak, so a breakout bar after a trend is not a "new" regime.
    cooldown:
        A pre-built RegimeCooldown (e.g. with custom families); when given it
        takes precedence over `min_stable_periods`.
    """
    mode = f"strategy={strategy.name} (pinned)" if strategy is not None else "strategy=<routed>"
    print(f"[{datetime.now()}] Orchestrator: symbol={symbol}, {mode}")
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

    # 1) detect regime, check it has settled, and route it to a strategy
    regime_info = detect_regime(df, detector)
    regime = regime_info['regime'] if regime_info else None
    print("Regime:", regime)

    cooldown = cooldown or RegimeCooldown(min_stable_periods)
    stability = cooldown.assess(regime_history(df, detector))
    print("Regime stability:", stability)

    router = router or StrategyRouter()
    decision = router.route(regime_info)
    print("Routing:", decision)

    position = broker.get_positions().get(symbol)
    in_position = bool(position) and float(position.get('qty', 0)) > 0
    price = float(df['Close'].iloc[-1])

    routed = strategy is None
    plan = plan_cycle(decision, symbol, in_position, router,
                      pinned=strategy, force_flatten=force_flatten, stability=stability)

    # 2) regime discipline: the old regime's risk comes off *before* anything
    # else happens.  If the sell does not fully fill, stop here -- allocating
    # fresh capital on top of residual exposure is exactly what we must avoid.
    ledger = ledger if ledger is not None else SettlementLedger()
    if plan.flatten is not None:
        result = flatten_position(plan.flatten, symbol, broker, position, price,
                                  plan.flatten_reason, ledger=ledger)
        if not result.closed:
            print("Cycle halted: not allocating to a new strategy while the old position is still open.")
            return
        position, in_position = None, False
        print(f"Liquidation of {result.liquidation.qty} {symbol} "
              f"(~{result.liquidation.proceeds:.2f}) is treated as UNSETTLED for the rest of this cycle: "
              f"it still counts toward exposure and its cash is not available for sizing.")

    strategy = plan.strategy
    if strategy is None:
        print(f"No strategy for this cycle: {plan.reason}. Done.")
        return
    print(f"Using strategy {strategy.name}: {plan.reason}")

    # 3) generate signals
    sig_df = strategy.generate_signals(df)
    latest = sig_df.iloc[-1]
    signal = int(latest.get('signal', 0))
    print("Latest signal:", signal)

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

    # 4) strategy health gate
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

    # 5) regime gate (pinned strategies only -- in routed mode the router *is*
    # the regime policy, and a custom map or explicit fallback must not be
    # second-guessed by the registry-derived table here)
    if not routed:
        allowed = REGIME_MAP.get(strategy.name, {}).get('allowed', [])
        if allowed and (regime not in allowed):
            print(f"Regime gate failed: {regime} not in {allowed}")
            return

    # 6) risk sizing: MAX_EXPOSURE_PER_TRADE of *total portfolio value*
    # (cash + open positions at market), capped so notional + fees fit in
    # *settled* cash -- proceeds of any unsettled liquidation are excluded.
    snapshot = compute_portfolio_exposure(broker, {symbol: price}, pending=ledger)
    if snapshot['unsettled_proceeds'] > 0 or snapshot['in_flight_value'] > 0:
        print(f"Unsettled liquidations: {len(ledger)} sale(s), proceeds {snapshot['unsettled_proceeds']:.2f} "
              f"withheld from cash ({snapshot['cash']:.2f} -> settled {snapshot['settled_cash']:.2f}), "
              f"in-flight exposure {snapshot['in_flight_value']:.2f} still counted")
    fee_pct = estimated_fee_pct_for(broker)
    qty = compute_position_qty(snapshot['portfolio_value'], price,
                               exposure_frac=MAX_EXPOSURE_PER_TRADE,
                               cash_available=snapshot['settled_cash'], fee_pct=fee_pct)
    if qty <= 0:
        print(f"Sizing produced 0 shares (portfolio {snapshot['portfolio_value']:.2f}, "
              f"settled cash {snapshot['settled_cash']:.2f}, price {price}). Skipping.")
        return
    print(f"Sized {qty} shares = {qty * price:.2f} "
          f"({qty * price / snapshot['portfolio_value']:.2%} of portfolio, target {MAX_EXPOSURE_PER_TRADE:.0%})")

    # 7) portfolio exposure gate: per-trade sizing is not enough on its own --
    # the sum of every open position, plus unsettled liquidations still in
    # flight, plus this order must stay within MAX_TOTAL_EXPOSURE of portfolio
    # value, otherwise the signal is skipped.
    allowed, exp = check_total_exposure(broker, symbol, qty, price,
                                        max_total_exposure=MAX_TOTAL_EXPOSURE, snapshot=snapshot)
    print(f"Exposure: current {exp['exposure']:.1%} "
          f"(positions {exp['positions_value']:.2f} + in-flight {exp['in_flight_value']:.2f} "
          f"/ portfolio {exp['portfolio_value']:.2f}), "
          f"projected with this trade {exp['projected_exposure']:.1%}, limit {exp['max_total_exposure']:.0%}")
    if not allowed:
        print(f"Total exposure gate failed: {exp['projected_exposure']:.1%} would exceed "
              f"{exp['max_total_exposure']:.0%}. Skipping trade.")
        return

    # 8) place order
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
        'reason': REASON_SIGNAL,
    }
    append_trade_to_history(rec)
    print("Entry logged (paper).")
    return rec


def _exit_position(strategy: Strategy, symbol: str, broker: FyersPaperBroker,
                   position: Dict[str, Any], ref_price: float,
                   reason: str = REASON_SIGNAL):
    """
    Close the full open position, compute realised P&L from entry vs exit price
    (net of entry + exit fees) and log it so it feeds the strategy's metrics.
    `reason` is recorded in the trade log (signal exit vs forced flatten).
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
        'reason': reason,
    }
    append_trade_to_history(rec)
    print(f"Exit logged (paper, reason={reason}). Realised P&L: {pnl:.2f} (fees {fees:.2f})")
    return rec

# ---------- sample run ----------
if __name__ == "__main__":
    broker = FyersPaperBroker(
        initial_cash=200000.0,
        market_price_provider=default_price_provider,
        fees_pct=0.05,
    )
    # Let the regime decide which strategy trades.  Build the router once and
    # reuse it across symbols/cycles so strategy instances are shared.
    router = StrategyRouter()
    orchestrate_one_cycle(None, "RELIANCE.NS", broker, router=router)

    # To pin a single strategy instead (e.g. while evaluating it in paper mode):
    # orchestrate_one_cycle(get_strategy_class("Donchian_Breakout")(), "RELIANCE.NS", broker)