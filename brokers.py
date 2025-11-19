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
    """

    def __init__(self,
                 initial_cash: float = 100000.0,
                 market_price_provider: Optional[Callable[[str], float]] = None,
                 fees_pct: float = 0.05):
        self.cash = float(initial_cash)
        self.positions: Dict[str, Dict[str, Any]] = {}  # symbol -> {"qty": int, "avg_price": float}
        self.trade_log: list[Dict[str, Any]] = []
        self.fees_pct = float(fees_pct)  # percent e.g., 0.05 = 0.05%
        self._price_provider = market_price_provider or default_price_provider

    def place_order(self, symbol: str, side: str, qty: float,
                    order_type: str = "market", price: Optional[float] = None) -> Dict[str, Any]:
        side = side.upper()
        assert side in ("BUY", "SELL"), "side must be BUY or SELL"
        # Determine fill price
        fill_price = price if price is not None else self._price_provider(symbol)
        if fill_price is None or math.isnan(fill_price):
            return {"status": "rejected", "reason": "no-price", "symbol": symbol}

        qty = float(qty)
        # compute fees = fees_pct (percent) of notional per side
        fee = (self.fees_pct / 100.0) * (fill_price * qty)

        if side == "BUY":
            total_cost = fill_price * qty + fee
            if total_cost > self.cash + 1e-9:
                return {"status": "rejected", "reason": "insufficient-cash", "required": total_cost, "available": self.cash}
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
            if not pos or pos["qty"] < qty - 1e-9:
                return {"status": "rejected", "reason": "insufficient-qty", "available_qty": pos["qty"] if pos else 0}
            proceed = fill_price * qty
            fee = (self.fees_pct / 100.0) * (fill_price * qty)
            self.cash += proceed - fee
            pos["qty"] -= qty
            if pos["qty"] <= 0:
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
        side = side.upper()
        assert side in ("BUY", "SELL")
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