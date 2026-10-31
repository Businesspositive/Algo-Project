from .base import DataSource

__all__ = [
    "DataSource",
    "FyersDataSource",
    "YFinanceDataSource",
]


def __getattr__(name):
    if name == "FyersDataSource":
        from .fyers import FyersDataSource
        return FyersDataSource

    if name == "YFinanceDataSource":
        from .yfinance import YFinanceDataSource
        return YFinanceDataSource

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")