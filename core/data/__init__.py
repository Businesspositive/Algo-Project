"""
Data access layer: concrete sources implement the DataSource interface.
"""

from .base import DataSource
from .fyers import FyersDataSource
from .yfinance import YFinanceDataSource

__all__ = ["DataSource", "FyersDataSource", "YFinanceDataSource"]


