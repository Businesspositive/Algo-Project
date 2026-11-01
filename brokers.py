from __future__ import annotations
import time
import math
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional
import pandas as pd

# Optional external libs (Binance)
try:
    from binance.client import Client as BinanceClient
except Exception:
    BinanceClient = None

# ---------------------------
# Broker Interface
# ---------------------------
class BrokerInterface(ABC):
    @abstractmethod
    def place_order(self, symbol: str, side: str, qty: float,
                    order_type: str = "market", price: Optional[float] = None) -> Dict[str, Any]:
        """Place an order. side: 'BUY' or 'SELL'."""
        raise NotImplementedError

    @abstractmethod
    def get_balance(self) -> Dict[str, float]:
        """Return cash/balances."""
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Return current positions keyed by symbol."""
        raise NotImplementedError

    @abstractmethod
    def portfolio_value(self, price_lookup: Optional[Dict[str, float]] = None) -> float:
        """Return total portfolio value (cash + positions at given prices)."""
        raise NotImplementedError

# ---------------------------
# FyersPaperBroker (simulator)
# ---------------------------
class FyersPaperBroker(BrokerInterface):
    """
    Simulated paper broker for Indian equities.
    - market_price_provider(symbol) -> float should be provided (callable).
    - fees_pct: percent as 0.05 for 0.05% per executed side (you can include brokerage + taxes).
    - allow_fractional: Indian cash equities trade in whole shares, so fractional
      quantities are rejected by default.

    Every order attempt is validated the way a real broker would validate it and
    either fills or comes back as {"status": "rejected", "reason": <code>, ...}.
    The broker never raises for a bad order; rejection reasons are:
        invalid-side, invalid-qty, fractional-qty, invalid-symbol,
        unsupported-order-type, no-price, limit-not-marketable,
        insufficient-cash, insufficient-qty
    """

    SUPPORTED_ORDER_TYPES = ("market", "limit")

    def __init__(self,
                 initial_cash: float = 100000.0,
                 market_price_provider: Optional[Callable[[str], float]] = None,
                 fees_pct: float = 0.05,
                 allow_fractional: bool = False):
        self.cash = float(initial_cash)
        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol -> {"qty": int, "avg_price": float}
        self.trade_log: list[Dict[str, Any]] = []        # executed fills only
        self.rejections: list[Dict[str, Any]] = []       # audit trail of rejected attempts
        self.fees_pct = float(fees_pct)  # percent e.g., 0.05 = 0.05%
        self.allow_fractional = bool(allow_fractional)
        self._price_provider = market_price_provider or default_price_provider

    # ---- helpers -------------------------------------------------------
    def _reject(self, reason: str, symbol: str, side: str, qty: Any, **extra: Any) -> Dict[str, Any]:
        res = {"status": "rejected", "reason": reason, "symbol": symbol, "side": side, "qty": qty, **extra}
        self.rejections.append({**res, "time": time.time()})
        return res

    def _market_price(self, symbol: str) -> tuple[Optional[float], Optional[str]]:
        """Return (price, error). Provider failures are reported, never raised."""
        try:
            px = self._price_provider(symbol)
        except Exception as exc:
            return None, f"price-provider-error: {exc!r}"
        try:
            px = float(px) if px is not None else None
        except (TypeError, ValueError):
            return None, f"non-numeric price {px!r}"
        if px is None or math.isnan(px) or math.isinf(px) or px <= 0:
            return None, f"invalid price {px!r}"
        return px, None

    def _fee(self, price: float, qty: float) -> float:
        # fees = fees_pct (percent) of notional per side
        return (self.fees_pct / 100.0) * (price * qty)

    # ---- order entry ---------------------------------------------------
    def place_order(self, symbol: str, side: str, qty: float,
                    order_type: str = "market", price: Optional[float] = None) -> Dict[str, Any]:
        side_in = side
        side = str(side).upper().strip() if side is not None else ""
        if side not in ("BUY", "SELL"):
            return self._reject("invalid-side", symbol, side_in, qty, detail="side must be BUY or SELL")

        if not isinstance(symbol, str) or not symbol.strip():
            return self._reject("invalid-symbol", symbol, side, qty, detail="symbol must be a non-empty string")

        # quantity validation
        try:
            q = float(qty)
        except (TypeError, ValueError):
            return self._reject("invalid-qty", symbol, side, qty, detail="qty must be numeric")
        if math.isnan(q) or math.isinf(q) or q <= 0:
            return self._reject("invalid-qty", symbol, side, qty, detail="qty must be a positive finite number")
        if not self.allow_fractional and abs(q - round(q)) > 1e-9:
            return self._reject("fractional-qty", symbol, side, qty, detail="whole-share quantities only")
        qty = q

        order_type = str(order_type).lower().strip()
        if order_type not in self.SUPPORTED_ORDER_TYPES:
            return self._reject("unsupported-order-type", symbol, side, qty,
                                detail=f"supported: {self.SUPPORTED_ORDER_TYPES}")

        # Determine fill price
        if order_type == "market":
            if price is not None:
                # explicit fill-price override (handy for deterministic simulations)
                try:
                    fill_price = float(price)
                except (TypeError, ValueError):
                    return self._reject("no-price", symbol, side, qty, detail=f"non-numeric price {price!r}")
                if math.isnan(fill_price) or math.isinf(fill_price) or fill_price <= 0:
                    return self._reject("no-price", symbol, side, qty, detail=f"invalid price {price!r}")
            else:
                fill_price, err = self._market_price(symbol)
                if fill_price is None:
                    return self._reject("no-price", symbol, side, qty, detail=err)
        else:  # limit
            try:
                limit_price = float(price) if price is not None else None
            except (TypeError, ValueError):
                limit_price = None
            if limit_price is None or math.isnan(limit_price) or limit_price <= 0:
                return self._reject("no-price", symbol, side, qty, detail="limit orders require a positive price")
            market, err = self._market_price(symbol)
            if market is None:
                return self._reject("no-price", symbol, side, qty, detail=err)
            # no resting order book in the simulator: a limit order either
            # crosses the current market and fills at market, or is rejected.
            marketable = market <= limit_price if side == "BUY" else market >= limit_price
            if not marketable:
                return self._reject("limit-not-marketable", symbol, side, qty,
                                    limit_price=limit_price, market_price=market)
            fill_price = market

        fee = self._fee(fill_price, qty)

        if side == "BUY":
            total_cost = fill_price * qty + fee
            if total_cost > self.cash + 1e-9:
                return self._reject("insufficient-cash", symbol, side, qty,
                                    required=total_cost, available=self.cash, price=fill_price)
            # update position
            pos = self.positions.get(symbol)
            if pos:
                new_qty = pos["qty"] + qty
                new_avg = (pos["avg_price"] * pos["qty"] + fill_price * qty) / new_qty
                pos["qty"] = new_qty
                pos["avg_price"] = new_avg
            else:
                self.positions[symbol] = {"qty": qty, "avg_price": fill_price}
            self.cash -= total_cost
            record = {"symbol": symbol, "side": side, "qty": qty, "price": fill_price, "fee": fee, "time": time.time()}
            self.trade_log.append(record)
            return {"status": "filled", "filled_qty": qty, "avg_fill_price": fill_price, "fee": fee}

        else:  # SELL
            pos = self.positions.get(symbol)
            available = pos["qty"] if pos else 0
            if not pos or pos["qty"] < qty - 1e-9:
                return self._reject("insufficient-qty", symbol, side, qty, available_qty=available)
            proceed = fill_price * qty
            self.cash += proceed - fee
            pos["qty"] -= qty
            if pos["qty"] <= 1e-9:
                del self.positions[symbol]
            record = {"symbol": symbol, "side": side, "qty": qty, "price": fill_price, "fee": fee, "time": time.time()}
            self.trade_log.append(record)
            return {"status": "filled", "filled_qty": qty, "avg_fill_price": fill_price, "fee": fee}

    def get_balance(self) -> Dict[str, float]:
        return {"cash": self.cash}

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        return self.positions.copy()

    def portfolio_value(self, price_lookup: Optional[Dict[str, float]] = None) -> float:
        total = self.cash
        for sym, pos in self.positions.items():
            price = (price_lookup.get(sym) if price_lookup else None) or self._price_provider(sym)
            total += pos["qty"] * (price or pos["avg_price"])
        return float(total)

# ---------------------------
# Default price provider (yfinance fallback)
# ---------------------------
def default_price_provider(symbol: str) -> Optional[float]:
    """
    Example fallback provider using yfinance. For production replace with Fyers live quote.
    """
    try:
        import yfinance as yf
        df = yf.download(symbol, period="2d", interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except Exception:
        return None

# ---------------------------
# Binance testnet implementation
# ---------------------------
class BinanceTestnetBroker(BrokerInterface):
    """
    Uses python-binance Client to place testnet orders.
    Make sure to set up Testnet keys and, for python-binance, set client.API_URL to testnet endpoint.
    """
    def __init__(self, api_key: str, api_secret: str, fees_pct: float = 0.04, testnet: bool = True):
        if BinanceClient is None:
            raise RuntimeError("python-binance is required for BinanceTestnetBroker. pip install python-binance")
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.fees_pct = float(fees_pct)
        self.client = BinanceClient(api_key, api_secret)
        if testnet:
            # python-binance requires setting API_URL to the testnet base
            try:
                self.client.API_URL = 'https://testnet.binance.vision/api'
            except Exception:
                pass

    def place_order(self, symbol: str, side: str, qty: float,
                    order_type: str = "market", price: Optional[float] = None) -> Dict[str, Any]:
        side = str(side).upper().strip()
        if side not in ("BUY", "SELL"):
            return {"status": "rejected", "reason": "invalid-side", "symbol": symbol, "side": side, "qty": qty}
        try:
            # MARKET order
            if order_type.lower() == "market":
                order = self.client.create_order(symbol=symbol, side=side, type='MARKET', quantity=qty)
                # order returns fills/avgPrice etc depending on exchange; compute filled
                filled_qty = sum(float(fill.get('qty', 0)) for fill in order.get('fills', [])) if order.get('fills') else float(order.get('executedQty', 0))
                avg_price = None
                if order.get('fills'):
                    notional = sum(float(fill['qty']) * float(fill['price']) for fill in order['fills'])
                    avg_price = notional / filled_qty if filled_qty > 0 else None
                else:
                    avg_price = float(order.get('fills', [{}])[0].get('price', 0)) if filled_qty else None
                fee = (self.fees_pct / 100.0) * (avg_price or 0) * (filled_qty or 0)
                return {"status": order.get('status'), "raw": order, "filled_qty": filled_qty, "avg_fill_price": avg_price, "fee": fee}
            else:
                # Limit orders, stop-limit etc. For test skeleton, fallback to market
                return {"status": "rejected", "reason": "only-market-supported-in-skeleton"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def get_balance(self) -> Dict[str, float]:
        try:
            bal = self.client.get_account()
            balances = {b['asset']: float(b['free']) + float(b['locked']) for b in bal['balances']}
            return balances
        except Exception as e:
            return {"error": str(e)}

    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        # For spot, positions = non-zero balances
        b = self.get_balance()
        if 'error' in b:
            return {}
        return {asset: {'qty': qty} for asset, qty in b.items() if asset != 'USDT' and qty > 0}

    def portfolio_value(self, price_lookup: Optional[Dict[str, float]] = None) -> float:
        balances = self.get_balance()
        if 'error' in balances:
            return 0.0
        # naive evaluation: sum assets valued in USDT if prices provided, else ignore
        total = 0.0
        for asset, qty in balances.items():
            if asset == 'USDT':
                total += qty
            else:
                pair = f"{asset}USDT"
                price = None
                if price_lookup and pair in price_lookup:
                    price = price_lookup[pair]
                else:
                    try:
                        tick = self.client.get_symbol_ticker(symbol=pair)
                        price = float(tick['price'])
                    except Exception:
                        price = 0.0
                total += qty * price
        return float(total)

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # ----- FYERS PAPER (simulated) -----
    # Example: using default_price_provider (yfinance) — replace with Fyers quote function
    fy = FyersPaperBroker(initial_cash=200000, market_price_provider=default_price_provider, fees_pct=0.05)
    print("Initial cash (FyersPaper):", fy.get_balance())

    # Simulate a buy of RELIANCE.NS
    res = fy.place_order("RELIANCE.NS", "BUY", qty=1)
    print("Fyers BUY result:", res)
    print("Positions:", fy.get_positions())
    print("Portfolio value:", fy.portfolio_value())

    # ----- BINANCE TESTNET (crypto) -----
    # You must provide testnet API keys
    # bin = BinanceTestnetBroker(api_key="YOUR_TESTNET_KEY", api_secret="YOUR_TESTNET_SECRET", fees_pct=0.04, testnet=True)
    # print("Binance balances:", bin.get_balance())
    # res2 = bin.place_order("BTCUSDT", "BUY", qty=0.0001)  # example
    # print("Binance order:", res2)