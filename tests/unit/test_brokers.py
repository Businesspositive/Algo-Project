from brokers import FyersPaperBroker


def test_paper_broker_buy_order(sample_ohlcv):
    broker = FyersPaperBroker(
        initial_cash=100000,
        market_price_provider=lambda symbol: 100.0,
        fees_pct=0,
    )

    result = broker.place_order("TEST", "BUY", 10)

    assert result["status"] == "filled"
    assert broker.get_balance()["cash"] == 99000
    assert broker.get_positions()["TEST"]["qty"] == 10


def test_paper_broker_sell_order():
    broker = FyersPaperBroker(
        initial_cash=100000,
        market_price_provider=lambda symbol: 100.0,
        fees_pct=0,
    )

    broker.place_order("TEST", "BUY", 10)
    result = broker.place_order("TEST", "SELL", 5)

    assert result["status"] == "filled"
    assert broker.get_positions()["TEST"]["qty"] == 5

# ---------------------------------------------------------------------------
# Rejection simulation: a bad order must come back as status='rejected',
# must never raise, and must leave cash/positions untouched.
# ---------------------------------------------------------------------------
import math

import pytest


def make_broker(price=100.0, cash=1000.0, fees_pct=0.05, **kw):
    return FyersPaperBroker(
        initial_cash=cash,
        market_price_provider=lambda symbol: price,
        fees_pct=fees_pct,
        **kw,
    )


def assert_rejected(broker, result, reason):
    assert result["status"] == "rejected"
    assert result["reason"] == reason
    assert broker.get_balance()["cash"] == 1000.0
    assert broker.get_positions() == {}
    assert broker.trade_log == []
    assert broker.rejections[-1]["reason"] == reason


def test_buy_exceeding_cash_is_rejected():
    broker = make_broker()
    result = broker.place_order("TEST", "BUY", 20)  # 2000 notional vs 1000 cash
    assert_rejected(broker, result, "insufficient-cash")
    assert result["required"] == pytest.approx(2001.0)
    assert result["available"] == 1000.0


def test_buy_rejected_when_fees_push_cost_over_cash():
    broker = make_broker()  # 10 @ 100 = 1000 exactly, +0.5 fee -> unaffordable
    result = broker.place_order("TEST", "BUY", 10)
    assert_rejected(broker, result, "insufficient-cash")
    # but 9 shares (900 + 0.45 fee) is fine
    assert broker.place_order("TEST", "BUY", 9)["status"] == "filled"
    assert broker.get_balance()["cash"] == pytest.approx(1000 - 900 - 0.45)


@pytest.mark.parametrize("qty", [0, -5, float("nan"), float("inf"), "abc", None])
def test_invalid_quantity_is_rejected(qty):
    broker = make_broker()
    result = broker.place_order("TEST", "BUY", qty)
    assert_rejected(broker, result, "invalid-qty")


def test_negative_quantity_cannot_create_cash():
    """Regression: a negative BUY used to 'fill' and credit the account."""
    broker = make_broker()
    broker.place_order("TEST", "BUY", -5)
    assert broker.get_balance()["cash"] == 1000.0


def test_fractional_quantity_rejected_for_equities_by_default():
    broker = make_broker()
    assert_rejected(broker, broker.place_order("TEST", "BUY", 1.5), "fractional-qty")


def test_fractional_quantity_allowed_when_enabled():
    broker = make_broker(allow_fractional=True)
    assert broker.place_order("TEST", "BUY", 1.5)["status"] == "filled"
    assert broker.get_positions()["TEST"]["qty"] == 1.5


@pytest.mark.parametrize("side", ["HOLD", "", None, "buy "])
def test_invalid_side_is_rejected_not_raised(side):
    broker = make_broker()
    result = broker.place_order("TEST", side, 1)
    if side == "buy ":  # case/whitespace tolerant
        assert result["status"] == "filled"
    else:
        assert_rejected(broker, result, "invalid-side")


@pytest.mark.parametrize("symbol", ["", "   ", None, 123])
def test_invalid_symbol_is_rejected(symbol):
    broker = make_broker()
    assert_rejected(broker, broker.place_order(symbol, "BUY", 1), "invalid-symbol")


@pytest.mark.parametrize("price", [None, float("nan"), 0.0, -1.0, "n/a"])
def test_bad_market_price_is_rejected(price):
    broker = make_broker(price=price)
    assert_rejected(broker, broker.place_order("TEST", "BUY", 1), "no-price")


def test_price_provider_exception_is_rejected_not_raised():
    def boom(symbol):
        raise ConnectionError("quote service down")

    broker = FyersPaperBroker(initial_cash=1000.0, market_price_provider=boom, fees_pct=0.05)
    result = broker.place_order("TEST", "BUY", 1)
    assert_rejected(broker, result, "no-price")
    assert "quote service down" in result["detail"]


def test_unsupported_order_type_is_rejected():
    broker = make_broker()
    assert_rejected(broker, broker.place_order("TEST", "BUY", 1, order_type="stop"), "unsupported-order-type")


def test_limit_order_requires_price():
    broker = make_broker()
    assert_rejected(broker, broker.place_order("TEST", "BUY", 1, order_type="limit"), "no-price")


def test_limit_buy_below_market_does_not_fill():
    """Regression: a limit BUY used to fill at the limit price regardless of the market."""
    broker = make_broker(price=100.0)
    result = broker.place_order("TEST", "BUY", 1, order_type="limit", price=50.0)
    assert_rejected(broker, result, "limit-not-marketable")
    assert result["market_price"] == 100.0


def test_limit_buy_at_or_above_market_fills_at_market():
    broker = make_broker(price=100.0)
    result = broker.place_order("TEST", "BUY", 1, order_type="limit", price=120.0)
    assert result["status"] == "filled"
    assert result["avg_fill_price"] == 100.0  # price improvement, not the limit


def test_limit_sell_above_market_does_not_fill():
    broker = make_broker(price=100.0)
    broker.place_order("TEST", "BUY", 1)
    result = broker.place_order("TEST", "SELL", 1, order_type="limit", price=150.0)
    assert result["status"] == "rejected"
    assert result["reason"] == "limit-not-marketable"
    assert broker.get_positions()["TEST"]["qty"] == 1


def test_sell_more_than_held_is_rejected():
    broker = make_broker()
    broker.place_order("TEST", "BUY", 3)
    cash_after_buy = broker.get_balance()["cash"]
    result = broker.place_order("TEST", "SELL", 5)
    assert result["status"] == "rejected"
    assert result["reason"] == "insufficient-qty"
    assert result["available_qty"] == 3
    assert broker.get_balance()["cash"] == cash_after_buy
    assert broker.get_positions()["TEST"]["qty"] == 3


def test_sell_without_position_is_rejected():
    broker = make_broker()
    assert_rejected(broker, broker.place_order("TEST", "SELL", 1), "insufficient-qty")


def test_rejections_are_audited_separately_from_fills():
    broker = make_broker()
    broker.place_order("TEST", "BUY", 999)   # rejected
    broker.place_order("TEST", "BUY", 1)     # filled
    assert len(broker.rejections) == 1
    assert len(broker.trade_log) == 1
    assert broker.rejections[0]["reason"] == "insufficient-cash"