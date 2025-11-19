from backtesting import Strategy
from utils.rsi_utils import compute_rsi

class RSIMeanReversionBT(Strategy):
    period = 14
    lower = 30
    upper = 55

    def init(self):
        # Compute RSI using the shared function
        rsi_series = compute_rsi(self.data.Close, self.period)
        self.rsi = self.I(lambda: rsi_series, name="RSI")

    def next(self):
        rsi_val = self.rsi[-1]
        if rsi_val < self.lower and not self.position:
            self.buy()
        elif self.position and rsi_val > self.upper:
            self.position.close()
