import pandas as pd
from datetime import datetime, timedelta
from Indicators import resample, Fractal, Supertrend, ATR


"""
TimeFrame: 15 min
Indicator: Fractal, Supertrend, ATR
Strategy: Fractal Breakout with Supertrend Confirmation
This strategy uses fractals to identify potential breakout points and uses the Supertrend indicator for trend confirmation. The ATR is used to set stop-loss and take-profit levels.

Buy Condition:
1. A fractal top is formed, indicating a potential reversal point.
2. The price breaks above the fractal top.
3. The Supertrend indicator is in a bullish state (price is above the Supertrend line).

Sell Condition:
1. A fractal bottom is formed, indicating a potential reversal point.
2. The price breaks below the fractal bottom.
3. The Supertrend indicator is in a bearish state (price is below the Supertrend line).

Exit Conditions:
1. Stop-loss is triggered if the price moves against the position by a certain level (1-atr).
2. Take-profit is triggered if the price moves in favor of the position by a certain level (2-atr).

This strategy is designed to capture significant price movements while managing risk through the use of ATR for dynamic stop-loss and take-profit levels.



Tushar:
	80 fees
	pnl calculation in inr = pnl * 93

"""


## load data D:\CryptoSniper\Backtest\ethusd_merged.csv

script = "ethusd"

df = pd.read_csv(f"{script}_merged.csv", parse_dates=["date"])

df1 = df[df["date"] >= datetime(2025, 4, 1)]

dfx = resample(df, "15T")

dfx.
