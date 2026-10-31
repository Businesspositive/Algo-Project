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