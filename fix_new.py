import os
import pandas as pd
import numpy as np
import itertools
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
import traceback
from numba import jit, prange, njit
from multiprocessing import Pool, cpu_count, Manager

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

FINAL_SHEET = []


def resample_data(dff, time_frame):
    df = dff.copy()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    resampled = df.resample(time_frame, origin='start').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    resampled.replace(0, np.nan, inplace=True)
    resampled.dropna(subset=['high'], inplace=True)
    resampled.fillna(0, inplace=True)
    resampled.reset_index(inplace=True)
    return resampled


@jit(nopython=True, parallel=True)
def detect_fractals(highs, lows, dates_int, consecutive=5):
    lookback = (consecutive - 1) // 2
    n = (consecutive // 2) + 1
    fractal_top = np.full(len(lows), np.nan)
    fractal_bottom = np.full(len(lows), np.nan)

    fractal_time_top = np.full(len(highs), np.nan)
    fractal_time_bottom = np.full(len(highs), np.nan)

    for i in prange(lookback, len(highs) - lookback):
        is_fractal_top = True
        is_fractal_bottom = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_fractal_top = False
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_fractal_bottom = False

        if is_fractal_top:
            fractal_top[i] = highs[i]
            fractal_time_top[i] = dates_int[i + n]
        if is_fractal_bottom:
            fractal_bottom[i] = lows[i]
            fractal_time_bottom[i] = dates_int[i + n]

    return fractal_top, fractal_bottom, fractal_time_top, fractal_time_bottom


@njit
def SuperTrend_optimized(high, low, close, atr_period, factor):
    hl2, atr, upper_band, lower_band = calculate_bands(high, low, close, atr_period, factor)
    supertrend_values, direction = calculate_supertrend(close, upper_band, lower_band, atr)

    return supertrend_values, direction


@njit
def calculate_bands(high, low, close, atr_period, factor):
    """Calculate the HL2, ATR, Upper and Lower bands"""
    n = len(high)
    hl2 = np.zeros(n)
    tr = np.zeros(n)
    atr = np.zeros(n)
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)

    # Calculate HL2 (midpoint of high and low)
    for i in range(n):
        hl2[i] = (high[i] + low[i]) / 2

    # Calculate True Range
    for i in range(n):
        hl = high[i] - low[i]
        if i > 0:
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)
        else:
            tr[i] = hl

    # Calculate ATR using EMA
    alpha = 1.0 / atr_period
    atr[0] = tr[0]
    for i in range(1, n):
        if i >= atr_period:
            atr[i] = (alpha * tr[i]) + ((1 - alpha) * atr[i - 1])
        else:
            # Simple average for initial periods
            atr[i] = np.sum(tr[:i + 1]) / (i + 1)

    # Calculate upper and lower bands
    for i in range(n):
        upper_band[i] = hl2[i] + (factor * atr[i])
        lower_band[i] = hl2[i] - (factor * atr[i])

    return hl2, atr, upper_band, lower_band + 


@njit
def calculate_supertrend(close, upper_band, lower_band, atr):
    n = len(close)
    direction = np.zeros(n, dtype=np.int32)
    supertrend = np.zeros(n)
    final_upperband = np.copy(upper_band)
    final_lowerband = np.copy(lower_band)

    # First value
    if atr[0] == 0:
        direction[0] = 1
        supertrend[0] = final_upperband[0]
    else:
        direction[0] = 1
        supertrend[0] = final_upperband[0]

    for i in range(1, n):
        # Adjust lower band
        if (lower_band[i] > final_lowerband[i - 1]) or (close[i - 1] < final_lowerband[i - 1]):
            final_lowerband[i] = lower_band[i]
        else:
            final_lowerband[i] = final_lowerband[i - 1]

        # Adjust upper band
        if (upper_band[i] < final_upperband[i - 1]) or (close[i - 1] > final_upperband[i - 1]):
            final_upperband[i] = upper_band[i]
        else:
            final_upperband[i] = final_upperband[i - 1]

        # Determine direction
        if atr[i - 1] == 0:
            direction[i] = 1
        elif supertrend[i - 1] == final_upperband[i - 1]:
            if close[i] > final_upperband[i]:
                direction[i] = -1
            else:
                direction[i] = 1
        else:  # supertrend was the lower band
            if close[i] < final_lowerband[i]:
                direction[i] = 1
            else:
                direction[i] = -1

        # Set supertrend value
        if direction[i] == -1:
            supertrend[i] = final_lowerband[i]
        else:
            supertrend[i] = final_upperband[i]

    return supertrend, direction


@njit(parallel=True)
def find_entrydata(start, end, price, side, timestamps, highs, lows, closes):
    entryPrice = np.empty(len(start), dtype=np.float64)
    entryTime = np.empty(len(start), dtype=np.int64)

    for i in prange(len(start)):
        st = start[i]
        nd = end[i]
        pr = price[i]

        if side == 0:
            # Buy
            mask = (timestamps >= st) & (timestamps < nd) & (highs > pr)
        else:
            # Sell
            mask = (timestamps >= st) & (timestamps < nd) & (lows < pr)

        # Apply filter and find the first entry matching the conditions
        filtered_idx = np.where(mask)[0]

        if len(filtered_idx) > 0:
            idx = filtered_idx[0]
            entryPrice[i] = closes[idx]
            entryTime[i] = timestamps[idx]

    return entryPrice, entryTime


def df_normalizatoin(tf, fractal, atr_period, factor, df, dir_name):
    highs = df['high'].values
    lows = df['low'].values
    dates = df['date'].values
    closes = df['close'].values
    dates_int = np.array(dates, dtype='datetime64[s]').astype(int)

    fractal_top, fractal_bottom, fractal_time_top, fractal_time_bottom = detect_fractals(highs, lows, dates_int, consecutive=fractal)

    atr = calculate_atr(high=highs, low=lows, close=closes, window=14)

    supertrend_values, direction = SuperTrend_optimized(highs, lows, closes, atr_period, factor)
    df['SuperTrend'] = supertrend_values.round(3)
    df['Direction'] = direction

    df['fractal_top'] = fractal_top
    df['fractal_bottom'] = fractal_bottom
    df['fractalTime_Top'] = fractal_time_top
    df['fractalTime_Bottom'] = fractal_time_bottom
    df['atr'] = atr
    df['date'] = dates_int

    # # strategy setup

    df['fractal_bottom'].ffill(inplace=True)
    df['fractal_top'].ffill(inplace=True)
    df['fractalTime_Top'].ffill(inplace=True)
    df['fractalTime_Bottom'].ffill(inplace=True)
    df['p_atr'] = df['atr'].shift(1)
    df['Date'] = df['date'].shift(-1)
    df.dropna(axis=0, inplace=True)

    df['buy_bid'] = df['high'] + .1 * df['atr'].shift(-1)
    df['sell_bid'] = df['low'] - .1 * df['atr'].shift(-1)

    df.at[len(df) - 1, 'buy_bid'] = df.at[len(df) - 1, 'high'] + .1 * df.at[len(df) - 2, 'atr']
    df.at[len(df) - 1, 'sell_bid'] = df.at[len(df) - 1, 'low'] - .1 * df.at[len(df) - 2, 'atr']

    # df['Buy Qty'] = ( cap * leverage * df['buy_bid'] ) // cm
    # df['Sell Qty'] = ( cap * leverage * df['sell_bid'] ) // cm

    max_pnl_loss = -perTradLoss

    # buy_ep = 1 / df['buy_bid']
    # df['BuyMaxSL'] = (1 / (buy_ep - max_pnl_loss / (df['Buy Qty'] * cm))).round(2)

    # sell_ep = 1 / df['sell_bid']
    # df['SellMaxSL'] = (1 / (sell_ep - max_pnl_loss / (df['Sell Qty'] * -cm))).round(2)

    buy_entries = (df['high'] > df['fractal_top']) & (df['date'] > df['fractalTime_Top']) & (df['Direction'] == -1)
    sell_entries = (df['low'] < df['fractal_bottom']) & (df['date'] > df['fractalTime_Bottom']) & (df['Direction'] == 1)

    df['Buy Entry Cond'] = buy_entries
    df['Sell Entry Cond'] = sell_entries
    # # 1 min entry

    m_timestamps = min_df['timestamp'].values
    m_highs = min_df['high'].values
    m_lows = min_df['low'].values
    m_closes = min_df['close'].values

    start_date = df.loc[buy_entries, 'date'].to_numpy()
    end_date = df.loc[buy_entries, 'Date'].to_numpy()
    prices = df.loc[buy_entries, 'fractal_top'].to_numpy()

    # side 0 for buy and 1 for sell
    buyPrices , buyTimes = find_entrydata(start_date, end_date, prices, 0, m_timestamps, m_highs, m_lows, m_closes)
    df.loc[buy_entries, 'EntryPrice'] = buyPrices
    df.loc[buy_entries, 'EntryTime'] = buyTimes
    # # SELL
    start_date = df.loc[sell_entries, 'date'].to_numpy()
    end_date = df.loc[sell_entries, 'Date'].to_numpy()
    prices = df.loc[sell_entries, 'fractal_bottom'].to_numpy()

    sellPrices , sellTimes = find_entrydata(start_date, end_date, prices, 1, m_timestamps, m_highs, m_lows, m_closes)

    df.loc[sell_entries, 'EntryPrice'] = sellPrices
    df.loc[sell_entries, 'EntryTime'] = sellTimes

    main_df = df[buy_entries | sell_entries]
    dt = datetime(2022, 1, 1, 1, 0)
    checker = int(dt.timestamp())
    main_df = main_df[main_df['date'] >= checker]
    main_df = main_df.drop_duplicates(subset=['fractal_top', 'fractalTime_Top', 'Buy Entry Cond'], keep='first')
    entry_df = main_df.drop_duplicates(subset=['fractal_bottom', 'fractalTime_Bottom', 'Sell Entry Cond'], keep='first')

    p_atr = entry_df['p_atr'].values
    enp = entry_df['EntryPrice'].values
    ent = entry_df['EntryTime'].values
    buy_entries = entry_df['Buy Entry Cond'].values
    sell_entries = entry_df['Sell Entry Cond'].values
    # buy_max = entry_df['BuyMaxSL'].values
    # sell_max = entry_df['SellMaxSL'].values
    # bqty = entry_df['Buy Qty'].values
    # sqty = entry_df['Sell Qty'].values
    ft = entry_df['fractal_top'].values
    ftt = entry_df['fractalTime_Top'].values
    fb = entry_df['fractal_bottom'].values
    fbt = entry_df['fractalTime_Bottom'].values

    dir_name = f"{dir_name}/{tf}/{fractal}/{atr_period}/{factor}"
    # os.makedirs(os.path.dirname(dir_name), exist_ok=True)

    for tg in tg_range:
        os.makedirs(f"{dir_name}/{tg}", exist_ok=True)
        for sl in sl_range:
            key = f"{dir_name}/{tg}/Trades_{sl}_.csv"
            if os.path.isfile(key) and os.path.getsize(dir_name) > 10240:
                continue

            np_dtype = np.dtype(
                {
                    "names": [
                        "EntryTime",
                        "EntryPrice",
                        "Side",
                        "FratalPrice",
                        "FratalTime",
                        "ATR",
                        "SL ATR",
                        "SL",
                        "Target",
                        "ExitTime",
                        "ExitPrice",
                        "ExitType"
                    ],
                    "formats": [
                        np.int64,
                        np.float64,
                        np.int32,
                        np.float64,
                        np.float64,
                        np.float64,
                        np.float64,
                        np.float64,
                        np.float64,
                        np.int64,
                        np.float64,
                        np.int32     
                    ],
                }
            )

            trade_data = np.ones(entry_df.shape[0], dtype=np_dtype)

            result_data = backtest_new(
                enp, ent, buy_entries, sell_entries, p_atr, ft, ftt, fb, fbt, tg, sl,
                m_timestamps, m_highs, m_lows, m_closes, tf, fractal, trade_data
            )

            sheet = pd.DataFrame(result_data)
            sheet.sort_values(by=["EntryTime"], axis=0, inplace=True)
            sheet["Side"].replace({0.0: "Buy", 1.0: "Sell"}, inplace=True)
            sheet["ExitType"].replace({0: "SL", 1: "TG"}, inplace=True)

            buyindex = sheet[sheet["Side"] == "Buy"].index
            sellindex = sheet[sheet["Side"] == "Sell"].index

            sheet["PNL"] = 0
            sheet["fee"] = 0
            sheet["PNL With Expense"] = 0
            sheet["PNL_points"] = 0

            sheet.loc[sellindex, "QTY"] = (
                sheet["PNL"]  ((1 / sheet["EntryPrice"] - 1 / sheet["ExitPrice"]) - cm
                ))

            sheet.loc[buyindex, "QTY"] = (
                ((1 / sheet["EntryPrice"]) - (1 / sheet["ExitPrice"]))
                * sheet["QTY"]
                * -cm
            )

            sheet.loc[sellindex, "PNL"] = (
                ((1 / sheet["EntryPrice"]) - (1 / sheet["ExitPrice"]))
                * sheet["QTY"]
                * -cm
            )
            sheet.loc[sellindex, "fee"] = (
                ((1 / sheet["EntryPrice"]) + (1 / sheet["ExitPrice"]))
                * 0.0005
                * sheet["QTY"]
                * cm
            )
            sheet.loc[sellindex, "PNL With Expense"] = sheet["PNL"] - sheet["fee"]
            sheet.loc[sellindex, "PNL_points"] = sheet["EntryPrice"] - sheet["ExitPrice"]

            sheet.loc[buyindex, "PNL"] = (
                ((1 / sheet["EntryPrice"]) - (1 / sheet["ExitPrice"]))
                * sheet["QTY"]
                * cm
            )
            sheet.loc[buyindex, "fee"] = (
                ((1 / sheet["EntryPrice"]) + (1 / sheet["ExitPrice"]))
                * 0.0005
                * sheet["QTY"]
                * cm
            )
            sheet.loc[buyindex, "PNL With Expense"] = sheet["PNL"] - sheet["fee"]
            sheet.loc[buyindex, "PNL_points"] = sheet["ExitPrice"] - sheet["EntryPrice"]
            sheet["EntryTime"] = pd.to_datetime(sheet['EntryTime'], unit="s")
            sheet["FratalTime"] = pd.to_datetime(sheet['FratalTime'], unit="s")
            sheet["ExitTime"] = pd.to_datetime(sheet['ExitTime'], unit="s")
            print(key, sheet.shape)
            sheet.to_csv(key)

    # pdb.set_trace()
    # print(datetime.now())

    return True


@jit(nopython=True, parallel=True)
def calculate_atr(high, low, close, window=14):

    high_low = high - low
    high_prev_close = np.abs(high - np.roll(close, 1))
    low_prev_close = np.abs(low - np.roll(close, 1))

    true_range = np.maximum(np.maximum(high_low, high_prev_close), low_prev_close)

    atr = np.zeros(len(high))

    atr[:window] = np.mean(true_range[:window])

    for i in prange(window, len(high)):
        atr[i] = (atr[i - 1] * (window - 1) + true_range[i]) / window

    return atr


@jit(nopython=True)
def find_exit(m_timestamps, m_lows, m_highs, t, side, sl_val, tg_val):
    start_idx = 0
    for j in range(len(m_timestamps)):
        if m_timestamps[j] <= (t + 60):
            start_idx = j
            continue

    for j in range(start_idx, len(m_timestamps)):
        if side == 0:  # Buy
            if m_lows[j] <= sl_val or m_highs[j] >= tg_val:
                return m_timestamps[j], sl_val if m_lows[j] <= sl_val else tg_val, 0.0 if m_lows[j] <= sl_val else 1.0

        elif side == 1:  # Sell
            if m_highs[j] >= sl_val or m_lows[j] <= tg_val:
                return m_timestamps[j], sl_val if m_highs[j] >= sl_val else tg_val, 0.0 if m_highs[j] >= sl_val else 1.0

    return 0.0, 0.0, 2.0


@jit(nopython=True)
def backtest_new(entryPrices, entryTimes, buy_entries, sell_entries, p_atr, ft, ftt, fb, fbt, tg, sl, m_timestamps, m_highs, m_lows, m_closes, tf, fractal, trade_data):

    trade_count = 0 

    for i in range(len(entryPrices)):
        p = entryPrices[i]
        t = entryTimes[i]
        buy = buy_entries[i]
        sell = sell_entries[i]
        atr = p_atr[i]
        frac_top = ft[i]
        frac_top_t = ftt[i]
        frac_bottom = fb[i]
        frac_bottom_t = fbt[i]

        if buy:
            sl_val = p - sl * atr
            tg_val = p + tg * atr
            sl_check = sl_val  # max(sl_val, bmax)
            exitTime, exitPrice, exitType = find_exit(m_timestamps, m_lows, m_highs, t, side=0, sl_val=sl_check, tg_val=tg_val)

            if exitPrice is not None and exitType != 2.0:
                # Store trade data
                trade_data[trade_count]['EntryTime'] = t
                trade_data[trade_count]['EntryPrice'] = p
                trade_data[trade_count]['Side'] = 0.0  # Buy side
                # trade_data[trade_count]['QTY'] = bqty_val
                trade_data[trade_count]['FratalPrice'] = frac_top
                trade_data[trade_count]['FratalTime'] = frac_top_t
                trade_data[trade_count]['ATR'] = atr
                trade_data[trade_count]['SL ATR'] = sl_val
                # trade_data[trade_count]['MAX_SL'] = bmax
                trade_data[trade_count]['SL'] = sl_check
                trade_data[trade_count]['Target'] = tg_val
                trade_data[trade_count]['ExitTime'] = exitTime
                trade_data[trade_count]['ExitPrice'] = exitPrice
                trade_data[trade_count]['ExitType'] = exitType
                
                trade_count += 1

        elif sell:
            sl_val = p + sl * atr
            tg_val = p - tg * atr
            sl_check = sl_val  # min(sl_val, smax)
            exitTime, exitPrice, exitType = find_exit(m_timestamps, m_lows, m_highs, t, side=1, sl_val=sl_check, tg_val=tg_val)

            if exitPrice is not None and exitType != 2.0:
                # Store trade data
                trade_data[trade_count]['EntryTime'] = t
                trade_data[trade_count]['EntryPrice'] = p
                trade_data[trade_count]['Side'] = 1.0  # sell side
                # trade_data[trade_count]['QTY'] = sqty_val
                trade_data[trade_count]['FratalPrice'] = frac_bottom
                trade_data[trade_count]['FratalTime'] = frac_bottom_t
                trade_data[trade_count]['ATR'] = atr
                trade_data[trade_count]['SL ATR'] = sl_val
                # trade_data[trade_count]['MAX_SL'] = smax
                trade_data[trade_count]['SL'] = sl_check
                trade_data[trade_count]['Target'] = tg_val
                trade_data[trade_count]['ExitTime'] = exitTime
                trade_data[trade_count]['ExitPrice'] = exitPrice
                trade_data[trade_count]['ExitType'] = exitType

                trade_count += 1

    # Trim the trade data array to the actual size
    return trade_data


def process_backtest_results(key, raw_data):
    processed_data = []
    for row in raw_data:
        side = 'Buy' if row[3] == 0.0 else 'Sell'
        exit_type = 'SL' if row[14] == 0.0 else 'TG'

        processed_row = (
            row[1], row[2], side, row[4], row[5], row[6], row[7],
            row[8], row[9], row[10], row[11], row[12], row[13], exit_type
        )
        processed_data.append(processed_row)

    sheet = pd.DataFrame(processed_data, columns=[
        'EntryTime', 'EntryPrice', 'Side', 'QTY', 'FratalPrice', 'FratalTime', 'ATR',
        'SL ATR', 'MAX_SL', 'SL', 'Target', 'ExitTime', 'ExitPrice', 'ExitType'
    ])

    # print(key)
    sheet.sort_values(by=["EntryTime"], axis=0, inplace=True)

    buyindex = sheet[sheet["Side"] == "Buy"].index
    sellindex = sheet[sheet["Side"] == "Sell"].index

    sheet["PNL"] = 0
    sheet["fee"] = 0
    sheet["PNL With Expense"] = 0
    sheet["PNL_points"] = 0

    sheet["QTY"] = (perTradLoss * ((1 / sheet["EntryPrice"] - 1 / sheet["SL"]) * cm))

    sheet.loc[sellindex, "PNL"] = (
        ((1 / sheet["EntryPrice"]) - (1 / sheet["ExitPrice"]))
        * sheet["QTY"]
        * -cm
    )
    sheet.loc[sellindex, "fee"] = (
        ((1 / sheet["EntryPrice"]) + (1 / sheet["ExitPrice"]))
        * 0.0005
        * sheet["QTY"]
        * cm
    )
    sheet.loc[sellindex, "PNL With Expense"] = sheet["PNL"] - sheet["fee"]
    sheet.loc[sellindex, "PNL_points"] = sheet["EntryPrice"] - sheet["ExitPrice"]

    sheet.loc[buyindex, "PNL"] = (
        ((1 / sheet["EntryPrice"]) - (1 / sheet["ExitPrice"]))
        * sheet["QTY"]
        * cm
    )
    sheet.loc[buyindex, "fee"] = (
        ((1 / sheet["EntryPrice"]) + (1 / sheet["ExitPrice"]))
        * 0.0005
        * sheet["QTY"]
        * cm
    )
    sheet.loc[buyindex, "PNL With Expense"] = sheet["PNL"] - sheet["fee"]
    sheet.loc[buyindex, "PNL_points"] = sheet["ExitPrice"] - sheet["EntryPrice"]
    sheet["EntryTime"] = pd.to_datetime(sheet['EntryTime'], unit="s")
    sheet["FratalTime"] = pd.to_datetime(sheet['FratalTime'], unit="s")
    sheet["ExitTime"] = pd.to_datetime(sheet['ExitTime'], unit="s")

    sheet.to_csv(f"{key}.csv")

    return True


def Runner(timeFrame, Fractal, st_atr_range, st_factor):
    # res_df = resample_data(dff=min_df, time_frame=f"{timeFrame}min")
    df = dict_manager.get(f'{timeFrame}')
    # dict_manager[f"{timeFrame}"]=res_df

    df_normalizatoin(timeFrame, Fractal, st_atr_range, st_factor, df, dir_name)


def calculate_drawdown(dff):
    df = dff.copy(deep=True)
    df['EntryTime'] = pd.to_datetime(df['EntryTime'])
    df = df.sort_values(by='EntryTime')
    df['Cumulative_PNL'] = df['PNL With Expense'].cumsum()
    df['Rolling_Max'] = df['Cumulative_PNL'].cummax()
    df['Drawdown'] = df['Cumulative_PNL'] - df['Rolling_Max']
    df['Drawdown'] = pd.to_numeric(df['Drawdown'], errors='coerce')
    max_drawdown = df['Drawdown'].min()

    drawdown_df = df[np.isclose(df['Drawdown'], max_drawdown)]
    end_date = df[df['Drawdown'] == max_drawdown].iloc[0]['EntryTime'].date()
    start_date = df[(df.EntryTime <= pd.to_datetime(end_date)) & (df['Drawdown'] >= 0)].iloc[-1]['EntryTime'].date()

    if not drawdown_df.empty:
        end_date = drawdown_df.iloc[0]['EntryTime'].date()
    else:
        end_date = df.iloc[0]['EntryTime'].date()

    return max_drawdown, start_date, end_date


def generate_sheet(tf, ft, atr_p, factor, tg):
    results = []
    for ss in list(np.arange(0.75, 2.25, 0.25)):
        try:
            dir_name = f"./SPT_NEW/{script}/{tf}/{ft}/{atr_p}/{factor}/{tg}/Trades_{ss}_.csv"

            if os.path.exists(dir_name):
                file_df = pd.read_csv(dir_name)
                if len(file_df) < 2:
                    print(dir_name)
                    continue

                # file_df["EntryTime"] =  pd.to_datetime(file_df['EntryTime'],unit="s")
                # file_df["FratalTime"] =  pd.to_datetime(file_df['FratalTime'],unit="s")
                # file_df["ExitTime"] =  pd.to_datetime(file_df['ExitTime'],unit="s")
                pnl = file_df['PNL'].sum()
                pnl_exp = file_df['PNL With Expense'].sum()
                nLoss = len(file_df[file_df['ExitType'] == "SL"])
                nProfit = len(file_df[file_df['ExitType'] == "TG"])
                max_drawdown, start_date, end_date = calculate_drawdown(dff=file_df)
                
                total_trades = len(file_df)
                # file_df.to_csv(f"./SPT/{script}/{tf}_{ft}_{atr_p}_{factor}_{tg}_Trades_{ss}_.csv")
                dct = {
                    "TimeFrame": tf,
                    "Fractal": ft,
                    "SPT ATR":atr_p,
                    "SPT FACTOR":factor,
                    "Target ATR": tg,
                    "SL ATR": ss,
                    "PNL": pnl,
                    "PNL With Expense": pnl_exp,
                    "Total Trade": total_trades,
                    "Positive Trade": nProfit,
                    "Negative Trade": nLoss,
                    "Accuracy": (nProfit / total_trades),
                    "Drawdown": max_drawdown,
                    "Start Date": start_date,
                    "End Date": end_date
                }
                results.append(dct)
            else:
                continue

        except Exception as e:
            print(f"Error processing {dir_name}: {e}")
            traceback.print_exc()

    return results


if __name__ == "__main__":

    script = "btcusd"  # "ethusd"
    tf_range = list(range(15, 101))
    ft_range = list(range(5, 13, 2))
    st_atr_range = np.arange(10.0, 21, 1)
    st_factor = np.arange(1.0, 6.0, 1)
    tg_range = np.array([1, 2, 3], dtype=np.float64)
    sl_range = np.array(np.arange(0.75, 2.25, 0.25), dtype=np.float64)

    wb = 1.2
    cap = 0.01
    leverage = 25
    cm = 100
    perTradLoss = 0.01
    num_cores = 25

    # tf_range =[15]
    # ft_range = [5]
    # st_atr_range = [16.0]
    # st_factor  = [2.0]
    # tg_range = [1.0]
    # sl_range =[1.0]

    script_df = pd.read_csv(f"{script}_merged.csv", parse_dates=['date'])
    min_df = script_df[script_df['date'] >= datetime(2021, 12, 30)].copy(deep=True)
    min_dates = np.array(min_df['date'].values, dtype='datetime64[s]').astype(int)
    min_df['timestamp'] = min_dates
    # #
    dict_manager = Manager().dict()

    # param_combinations = []

    # for x in range(15,101):
    #     for ft in range(5,13,2):
    #         for spt_atr in st_atr_range:
    #             for spt_f in st_factor:
    #                 for tg in [1.0,2.0,3.0]:
    #                     for sl in [0.75,1.0,1.25,1.5,1.75,2.0]:
    #                         filePath = f"/media/indian_live/1 tb/Binance_VP/SPT_NEW/{script}/{x}/{ft}/{spt_atr}/{spt_f}/{tg}/Trades_{sl}_.csv"
    #                         if os.path.getsize(filePath) < 10240:
    #                             # 
    #                             param_combinations.append((x, ft, spt_atr, spt_f))
    #                             res_df = resample_data(dff=min_df, time_frame=f"{x}min")
    #                             dict_manager[f"{x}"]=res_df

        # print(x)

    # print(len(param_combinations))

    # for tf in tf_range:
    #     res_df = resample_data(dff=min_df, time_frame=f"{tf}min")
    #     dict_manager[f"{tf}"]=res_df

    # # normalization
    dir_name = f"./SPT_NEW/{script}"
    os.makedirs(os.path.dirname(dir_name), exist_ok=True)

    # structured numpy dtype for the backtest data
    dtype = np.dtype([
        ('EntryTime', 'float64'),
        ('EntryPrice', 'float64'),
        ('Side', 'U4'),
        ('QTY', 'float64'),
        ('FratalPrice', 'float64'),
        ('FratalTime', 'float64'),
        ('ATR', 'float64'),
        ('SL ATR', 'float64'),
        ('MAX_SL', 'float64'),
        ('SL', 'float64'),
        ('Target', 'float64'),
        ('ExitTime', 'float64'),
        ('ExitPrice', 'float64'),
        ('ExitType', 'U2')
    ])

    # param_combinations = list(itertools.product(tf_range, ft_range,st_atr_range,st_factor))
    # dt = datetime.now().replace(second=0,microsecond=0)
    # print(f"Start {dt}")

    # pool_btst = Pool(processes=num_cores)
    # pool_btst.starmap(Runner, param_combinations)
    # pool_btst.close()
    # pool_btst.join()

    # dtt = datetime.now().replace(second=0,microsecond=0)
    # print(f"END {dtt}")

    # df_normalizatoin(15,5,dict_manager,dir_name)

    # tf_range =[18,19,20]
    # ft_range = [5,7]
    # tg_range = [2.0,3.0]
    # sl_range =[1.0,1.5,2.0]
    # st_atr_range = [15.0,16.0,19.0]
    # st_factor  = [2.0,3.0,4.0]

    combinations = list(itertools.product(tf_range, ft_range, st_atr_range, st_factor, tg_range))
    pool = Pool(processes=num_cores)
    results_list = pool.starmap(generate_sheet, [(tf, ft, atr_p, factor, tg) for tf, ft, atr_p, factor, tg in combinations])
    pool.close()
    pool.join()

    # Flatten the results
    all_results = [item for sublist in results_list for item in sublist]

    if all_results:
        fn = pd.DataFrame(all_results)
        fn.to_csv(f"{script}_FRACTAL_SPT_REPORT_NEW_.csv", index=False)

        print("CSV file saved.")
    else:
        print("No data to write to CSV.")

