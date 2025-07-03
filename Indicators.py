from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not all(
        col in df.columns for col in ["date", "open", "high", "low", "close", "volume"]
    ):
        raise ValueError(
            "DataFrame must contain 'date', 'open', 'high', 'low', 'close', 'volume' columns."
        )
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    original_last_timestamp = df["date"].max()

    df = df.set_index("date").sort_index()

    ohlcv_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    resampled_df = df.resample(timeframe, origin="start_day").agg(ohlcv_dict)

    resampled_df.dropna(subset=["close"], inplace=True)

    timeframe_delta = pd.to_timedelta(timeframe)

    resampled_df = resampled_df[
        resampled_df.index + timeframe_delta - pd.Timedelta(nanoseconds=1)
        <= original_last_timestamp
    ]

    return resampled_df.reset_index()


def detect_fractal_v2(df: pd.DataFrame, lookback_periods: int = 2) -> pd.DataFrame:

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    if lookback_periods < 1:
        raise ValueError("lookback_periods must be at least 1.")

    df_result = df.sort_values(by="date", ascending=True).copy()

    if df_result.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "fractal_top",
                "fractal_bottom",
                "fractal_time_top",
                "fractal_time_bottom",
            ]
        )

    highs = df_result["high"].values
    lows = df_result["low"].values
    dates = df_result["date"].values

    assert isinstance(
        len(highs), int
    ), f"len(highs) is not an integer! It's {type(len(highs))}"

    detected_fractal_top_price = np.full(len(highs), np.nan)
    detected_fractal_bottom_price = np.full(len(highs), np.nan)
    detected_fractal_top_time_list = [pd.NaT] * len(highs)
    detected_fractal_bottom_time_list = [pd.NaT] * len(highs)

    detected_fractal_top_time = np.array(detected_fractal_top_time_list)
    detected_fractal_bottom_time = np.array(detected_fractal_bottom_time_list)

    # --- Fractal Detection Loop ---
    for i in range(lookback_periods, len(highs) - lookback_periods):
        is_fractal_high = True
        for j in range(1, lookback_periods + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_fractal_high = False
                break

        if is_fractal_high:
            detected_fractal_top_price[i] = highs[i]
            detected_fractal_top_time[i] = dates[i]

        is_fractal_low = True
        for j in range(1, lookback_periods + 1):
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_fractal_low = False
                break

        if is_fractal_low:
            detected_fractal_bottom_price[i] = lows[i]
            detected_fractal_bottom_time[i] = dates[i]

    # Add temporary columns
    df_result["_temp_detected_fractal_top_price"] = detected_fractal_top_price
    df_result["_temp_detected_fractal_bottom_price"] = detected_fractal_bottom_price
    df_result["_temp_detected_fractal_top_time"] = detected_fractal_top_time
    df_result["_temp_detected_fractal_bottom_time"] = detected_fractal_bottom_time

    df_result["fractal_top"] = df_result["_temp_detected_fractal_top_price"].ffill()
    df_result["fractal_bottom"] = df_result[
        "_temp_detected_fractal_bottom_price"
    ].ffill()
    df_result["fractal_time_top"] = df_result["_temp_detected_fractal_top_time"].ffill()
    df_result["fractal_time_bottom"] = df_result[
        "_temp_detected_fractal_bottom_time"
    ].ffill()

    df_result["fractal_time_top"] = pd.to_datetime(df_result["fractal_time_top"])
    df_result["fractal_time_bottom"] = pd.to_datetime(df_result["fractal_time_bottom"])

    df_result["fractal_top"] = df_result["fractal_top"].shift(2)
    df_result["fractal_bottom"] = df_result["fractal_bottom"].shift(2)
    df_result["fractal_time_top"] = df_result["fractal_time_top"].shift(2)
    df_result["fractal_time_bottom"] = df_result["fractal_time_bottom"].shift(2)

    # Drop temporary columns
    df_result = df_result.drop(
        columns=[
            "_temp_detected_fractal_top_price",
            "_temp_detected_fractal_bottom_price",
            "_temp_detected_fractal_top_time",
            "_temp_detected_fractal_bottom_time",
        ]
    )

    return df_result


def ema(df, period):
    df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
    return df


def RSI(df, length=14):
    df = df.copy()
    change = df["close"].diff()

    up = np.where(change > 0, change, 0)
    down = np.where(change < 0, -change, 0)

    roll_up = pd.Series(up, index=df.index).ewm(alpha=1 / length, adjust=False).mean()
    roll_down = (
        pd.Series(down, index=df.index).ewm(alpha=1 / length, adjust=False).mean()
    )

    rs = roll_up / roll_down
    df["rsi"] = 100 - (100 / (1 + rs))

    return df


def atr(df):
    atr_period = 14
    df["high-low"] = df["high"] - df["low"]
    df["high-PrevClose"] = abs(df["high"] - df["close"].shift(1))
    df["low-PrevClose"] = abs(df["low"] - df["close"].shift(1))
    df["TrueRange"] = df[["high-low", "high-PrevClose", "low-PrevClose"]].max(axis=1)

    # df['ATR'] = df['TrueRange'].rolling(window=atr_period).mean()

    df["ATR"] = pine_rma(df, "TrueRange", 14)
    # df.drop(['high-low', 'high-PrevClose', 'low-PrevClose', 'TrueRange'], axis=1, inplace=True)

    return df


def pine_rma(df, src_column, length):
    alpha = 1 / length
    result = pd.Series(index=df.index, dtype=float)  # Initialize with NaN

    # Calculate the first value using the simple moving average (SMA) for the first period
    result.iloc[length - 1] = df[src_column].iloc[:length].mean()

    # Apply the recursive formula for the remaining values
    for i in range(length, len(df)):
        result.iloc[i] = (
            alpha * df[src_column].iloc[i] + (1 - alpha) * result.iloc[i - 1]
        )

    return result


def BB(df, period=20, sd=2):

    df["MB"] = df["close"].rolling(period).mean()
    std_value = df["close"].rolling(period).std(ddof=0)
    df["UB"] = df["MB"] + (sd * (std_value))
    df["LB"] = df["MB"] - (sd * (std_value))

    return None


def ichimoku(
    df, conversionPeriods=9, basePeriods=26, laggingSpan2Periods=52, displacement=26
):

    test = df.copy(deep=True)

    def donchian(test, length):
        return (
            test["high"].rolling(window=length).max()
            + test["low"].rolling(window=length).min()
        ) / 2

    test["BlueLine"] = donchian(test, conversionPeriods)  # blue line
    test["RedLine"] = donchian(test, basePeriods)  # red line
    test["leadLine1"] = ((test["BlueLine"] + test["RedLine"]) / 2).shift(
        -26
    )  # light green
    test["leadLine2"] = (donchian(test, laggingSpan2Periods)).shift(-26)  # light pink
    test["Gray"] = test["leadLine1"].shift(displacement - 1)
    test["Pink"] = test["leadLine2"].shift(displacement - 1)

    return test


def VWAP(df):

    len_df = df.shape[0]
    sumSrcVol = np.zeros(len_df)
    sumVol = np.zeros(len_df)
    vwap = np.zeros(len_df)

    src = (df["high"] + df["close"] + df["low"]) / 3
    src = src.to_numpy()
    vol = df["volume"].to_numpy()

    day_start = df[df["date"].dt.time == time(9, 15)].index

    other_time = df[~df.index.isin(day_start)].index

    for i in day_start:
        sumSrcVol[i] = src[i] * vol[i]
        sumVol[i] = vol[i]

    for i in other_time:
        sumSrcVol[i] = (src[i] * vol[i]) + sumSrcVol[i - 1]
        sumVol[i] = vol[i] + sumVol[i - 1]

    vwap = sumSrcVol / sumVol

    df["VWAP"] = vwap.round(2)

    return None


def detect_fractals(data, consecutive=5):
    """Vectorized fractal detection"""
    lookback = (consecutive - 1) // 2
    data = data.copy()
    n = (consecutive // 2) + 1

    if not pd.api.types.is_datetime64_any_dtype(data["date"]):
        data["date"] = pd.to_datetime(data["date"])

    highs = data["high"].values
    lows = data["low"].values
    dates = data["date"].values
    fractal_top = np.full(len(data), np.nan)
    fractal_bottom = np.full(len(data), np.nan)
    fractal_time_top = np.full(len(data), np.nan, dtype="datetime64[ns]")
    fractal_time_bottom = np.full(len(data), np.nan, dtype="datetime64[ns]")

    for i in range(lookback, len(data) - lookback - n):
        current_high = highs[i]
        current_low = lows[i]
        # date = dates[i]

        # Check previous and next candles
        prev_highs = highs[i - lookback : i]
        next_highs = highs[i + 1 : i + lookback + 1]
        prev_lows = lows[i - lookback : i]
        next_lows = lows[i + 1 : i + lookback + 1]

        # Fractal conditions
        if (current_high > prev_highs.max()) and (current_high > next_highs.max()):
            fractal_top[i] = current_high
            fractal_time_top[i] = dates[i + n]
        if (current_low < prev_lows.min()) and (current_low < next_lows.min()):
            fractal_bottom[i] = current_low
            fractal_time_bottom[i] = dates[i + n]

    data["fractal_top"] = fractal_top
    data["fractal_bottom"] = fractal_bottom
    data["fractalTime_Top"] = fractal_time_top
    data["fractalTime_Bottom"] = fractal_time_bottom

    return data


def SuperTrend(data, atrperiod, factor):

    y = data.copy(deep=True)

    y["date"] = y["date"].astype(str)
    hl = pd.Series(y["high"] - y["low"]).abs()
    hc = pd.Series(y["high"] - y["close"].shift()).abs()
    cl = pd.Series(y["close"].shift() - y["low"]).abs()
    hcl = pd.concat([hl, hc, cl], axis=1)
    tr = hcl.max(axis=1)
    y["hl2"] = (y["high"] + y["low"]) / 2
    y["atr"] = (
        tr.ewm(alpha=1 / atrperiod, min_periods=atrperiod).mean().round(7).fillna(0)
    )
    y["UB"] = y["hl2"] + factor * y["atr"]
    y["LB"] = y["hl2"] - factor * y["atr"]
    y["direction"] = 0
    y["supertrend"] = 0.0

    np_dtype = np.dtype({"names": y.dtypes.keys(), "formats": y.dtypes.values})
    x = np.ones(len(y), dtype=np_dtype)
    for i in y.dtypes.keys():
        x[i] = y[i].to_numpy()
    prevsupertrend = x["UB"][0]
    prevlowerband = x["LB"][0]
    prevupperband = x["UB"][0]
    for i in range(1, len(x)):
        if x["LB"][i] > prevlowerband or x["close"][i - 1] < prevlowerband:
            pass
        else:
            x["LB"][i] = prevlowerband
        if x["UB"][i] < prevupperband or x["close"][i - 1] > prevupperband:
            pass
        else:
            x["UB"][i] = prevupperband
        if x["atr"][i - 1] == 0:
            x["direction"][i] = 1
        elif prevsupertrend == prevupperband:
            if x["close"][i] > x["UB"][i]:
                x["direction"][i] = -1
            else:
                x["direction"][i] = 1
        else:
            if x["close"][i] < x["LB"][i]:
                x["direction"][i] = 1
            else:
                x["direction"][i] = -1
        if x["direction"][i] == -1:
            prevsupertrend = x["LB"][i]
        else:
            prevsupertrend = x["UB"][i]
        prevlowerband = x["LB"][i]
        prevupperband = x["UB"][i]

    return x["direction"]
