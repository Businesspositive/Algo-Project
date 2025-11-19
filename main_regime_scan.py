from datetime import datetime

from core.data import FyersDataSource
from screens.daily_selector import DailySelector


def main():
    # Define your stock universe
    # NIFTY50_UNIVERSE = [
    # "RELIANCE.NS"]
    NIFTY50_UNIVERSE = ["TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "LT.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "ASIANPAINT.NS",
    "BAJFINANCE.NS", "AXISBANK.NS", "MARUTI.NS", "HCLTECH.NS", "SUNPHARMA.NS",
    "ULTRACEMCO.NS", "WIPRO.NS", "ADANIGREEN.NS", "ONGC.NS", "TITAN.NS",
    "NESTLEIND.NS", "KOTAKBANK.NS", "POWERGRID.NS", "ADANIENT.NS", "HINDUNILVR.NS",
    "TECHM.NS", "TATAMOTORS.NS", "COALINDIA.NS", "GRASIM.NS", "BRITANNIA.NS",
    "TATASTEEL.NS", "JSWSTEEL.NS", "HDFCLIFE.NS", "DRREDDY.NS", "DIVISLAB.NS",
    "EICHERMOT.NS", "BAJAJFINSV.NS", "HEROMOTOCO.NS", "CIPLA.NS", "BPCL.NS",
    "HINDALCO.NS", "UPL.NS", "SBILIFE.NS", "SHREECEM.NS", "INDUSINDBK.NS",
    "APOLLOHOSP.NS", "BAJAJ-AUTO.NS", "M&M.NS", "TATACONSUM.NS", "ADANIPORTS.NS"
]

    data_source = FyersDataSource()
    selector = DailySelector(
        universe=NIFTY50_UNIVERSE,
        start="2025-01-01",
        data_source=data_source,
    )

    # Run pick for today's trades
    today = datetime.today().strftime("%Y-%m-%d")
    picks = selector.pick()

    print(f"\n=== Regime Scan for {today} ===")
    if not picks:
        print("No picks for today.")
        return

    for ticker, details in picks.items():
        print(f"{ticker:12} | Regime: {details['regime']:12}")
        for strat in details["strategies"]:
            print(
                f"  - {strat['name']:20} | signal={strat['signal']} | score={strat['score']:.2f}"
            )

if __name__ == "__main__":
    main()
