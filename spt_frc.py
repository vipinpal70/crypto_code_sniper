import pandas as pd
from datetime import datetime, timedelta
from Indicators import resample, detect_fractal_v2, SuperTrend, atr
import numpy as np
import pdb


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


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

df1 = df[df["date"] >= datetime(2025, 4, 1)].copy(deep=True)

dfx = resample(df, "15T")

# apply indicators

dfx = detect_fractal_v2(dfx, 2)
dfx["supertrend"] = SuperTrend(dfx, 10, 3.0)
atr(dfx)
dfx.dropna(inplace=True, axis=0)


dfx["p_atr"] = dfx["ATR"].shift(1)
dfx["p_supertrend"] = dfx["supertrend"].shift(1)
dfx["n_open"] = dfx["open"].shift(-1)
dfx["n_date"] = dfx["date"].shift(-1)

dfx = dfx[dfx["date"] >= datetime(2025, 6, 30)].copy(deep=True)

dfx["date"] = pd.to_datetime(dfx["date"], format="%Y-%m-%d %H:%M:%S")
dfx["n_date"] = pd.to_datetime(dfx["n_date"], format="%Y-%m-%d %H:%M:%S")

buy_entries = (
    (dfx["close"] > dfx["fractal_top"])
    & (dfx["date"] > dfx["fractal_time_top"])
    & (dfx["p_supertrend"].astype("Int32") == -1)
)

sell_entries = (
    (dfx["close"] < dfx["fractal_bottom"])
    & (dfx["date"] > dfx["fractal_time_bottom"])
    & (dfx["p_supertrend"].astype("Int32") == 1)
)


dfx["Buy Entry Cond"] = buy_entries
dfx["Sell Entry Cond"] = sell_entries
dfx.reset_index(inplace=True)

main_df = dfx.drop_duplicates(
    subset=["fractal_top", "fractal_time_top", "Buy Entry Cond"], keep="first"
)
entry_df = main_df.drop_duplicates(
    subset=["fractal_bottom", "fractal_time_bottom", "Sell Entry Cond"], keep="first"
)

entry_df.dropna(inplace=True, axis=0)
entry_df.reset_index(inplace=True)


## flags
prev_fractal = 0
prev_exit_time = entry_df.at[0, "date"]
EntryPrice = 0
entry = False
side = ""
final_sheet = []
# run_time = entry_df.at[0, "date"]


for i in range(len(entry_df) - 2):

    EntryTime = entry_df.at[i, "n_date"]

    if entry_df.at[i, "n_date"] < prev_exit_time:
        continue

    FractalTop = entry_df.at[i, "fractal_top"]
    FractalBottom = entry_df.at[i, "fractal_bottom"]
    FractalTopTime = entry_df.at[i, "fractal_time_top"]
    FractalBottomTime = entry_df.at[i, "fractal_time_bottom"]

    atr = entry_df.at[i, "p_atr"]
    spt = entry_df.at[i, "p_supertrend"]

    if entry_df.at[i, "Buy Entry Cond"] and FractalTopTime >= prev_exit_time:
        entry = True
        EntryPrice = entry_df.at[i, "n_open"]
        side = "buy"
        prev_fractal = FractalTop
        prev_exit_time = entry_df.at[i, "n_date"]

    if entry_df.at[i, "Sell Entry Cond"] and FractalBottomTime >= prev_exit_time:
        entry = True
        EntryPrice = entry_df.at[i, "n_open"]
        side = "sell"
        prev_fractal = FractalBottom
        prev_exit_time = entry_df.at[i, "n_date"]

    if entry and side == "buy":
        # print("Buy trade")
        sl = EntryPrice - (1 * atr)
        tg = EntryPrice + (2 * atr)
        sl_flt = dfx[(dfx["date"] > EntryTime) & (dfx["low"] <= sl)]
        tg_flt = dfx[(dfx["date"] > EntryTime) & (dfx["high"] >= tg)]

        if not sl_flt.empty and not tg_flt.empty:
            # check who is first
            first_sl = sl_flt.iloc[0]["date"]
            first_tg = tg_flt.iloc[0]["date"]

            if first_sl < first_tg:
                ExitPrice = sl
                ExitTime = first_sl
                ExitType = "SL"
            else:
                ExitPrice = tg
                ExitTime = first_tg
                ExitType = "TG"

        if sl_flt.empty and not tg_flt.empty:
            ExitPrice = tg
            ExitTime = tg_flt.iloc[0]["date"]
            ExitType = "TG"

        if tg_flt.empty and not sl_flt.empty:
            ExitPrice = sl
            ExitTime = sl_flt.iloc[0]["date"]
            ExitType = "SL"

        prev_exit_time = ExitTime
        doc = {
            "EntryTime": EntryTime,
            "EntryPrice": EntryPrice,
            "Side": side,
            "FractalPrice": FractalTop,
            "FractalTime": FractalTopTime,
            "SuperTrend": spt,
            "Atr": atr,
            "StopLoss": sl,
            "Target": tg,
            "ExitType": ExitType,
            "ExitPrice": ExitPrice,
            "ExitTime": ExitTime,
        }
        entry = False
        side = ""

        final_sheet.append(doc)
        # pdb.set_trace()

    if entry and side == "sell":
        # print("Sell trade")
        sl = EntryPrice + (1 * atr)  # SL above entry for sell
        tg = EntryPrice - (2 * atr)  # TG below entry for sell
        sl_flt = dfx[(dfx["date"] > EntryTime) & (dfx["high"] > sl)]
        tg_flt = dfx[(dfx["date"] > EntryTime) & (dfx["low"] < tg)]

        if not sl_flt.empty and not tg_flt.empty:
            # check who is first
            first_sl = sl_flt.iloc[0]["date"]
            first_tg = tg_flt.iloc[0]["date"]

            if first_sl < first_tg:
                ExitPrice = sl
                ExitTime = first_sl
                ExitType = "SL"
            else:
                ExitPrice = tg
                ExitTime = first_tg
                ExitType = "TG"

        if sl_flt.empty and not tg_flt.empty:
            ExitPrice = tg
            ExitTime = tg_flt.iloc[0]["date"]
            ExitType = "TG"

        if tg_flt.empty and not sl_flt.empty:
            ExitPrice = sl
            ExitTime = sl_flt.iloc[0]["date"]
            ExitType = "SL"

        prev_exit_time = ExitTime
        doc = {
            "EntryTime": EntryTime,
            "EntryPrice": EntryPrice,
            "Side": side,
            "FractalPrice": FractalTop,
            "FractalTime": FractalTopTime,
            "SuperTrend": spt,
            "Atr": atr,
            "StopLoss": sl,
            "Target": tg,
            "ExitType": ExitType,
            "ExitPrice": ExitPrice,
            "ExitTime": ExitTime,
        }
        entry = False
        side = ""

        final_sheet.append(doc)
        # pdb.set_trace()


sheet = pd.DataFrame(final_sheet)

sheet["pnl"] = np.where(
    sheet["Side"] == "buy",
    sheet["ExitPrice"] - sheet["EntryPrice"],
    sheet["EntryPrice"] - sheet["ExitPrice"],
)


sheet.to_csv("spt_frc.csv", index=False)
