import os
import traceback
import warnings
import time
from datetime import datetime, timedelta, timezone
import pytz
from Utils import file_path_locator, setup_logger
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import pymongo
from sys import argv
from os import path

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

load_dotenv()


def liveUpdate(STRATEGY: str):
    up = LiveCollection.find_one({"Strategy": STRATEGY})
    if up is None:
        logger.info("No live update found, fetching historical data")
        # get the previous hours time
        dt = datetime.now(timezone.utc) - timedelta(hours=1)
        setup_dict = {"Strategy": STRATEGY, "PREV FRACTAL": 0, "ExitTime": dt}
        LiveCollection.insert_one(setup_dict)
        return setup_dict if up is None else up

    return up


def ATR(df, atr_period):
    hl = pd.Series(df["high"] - df["low"]).abs()
    hc = pd.Series(df["high"] - df["close"].shift()).abs()
    cl = pd.Series(df["close"].shift() - df["low"]).abs()
    hcl = pd.concat([hl, hc, cl], axis=1)
    tr = hcl.max(axis=1)
    return tr.ewm(alpha=1 / atr_period, min_periods=atr_period).mean().round(2)


def detect_fractals(df, consecutive=5):
    highs = df["high"].values
    lows = df["low"].values
    date = df["date"].values

    lookback = (consecutive - 1) // 2
    n = (consecutive // 2) + 1
    fractal_top = np.full(len(highs), np.nan)
    fractal_bottom = np.full(len(highs), np.nan)

    # Initialize with explicit datetime64[ns] dtype
    fractal_time_top = np.full(len(highs), np.datetime64("NaT"), dtype="datetime64[ns]")
    fractal_time_bottom = np.full(
        len(highs), np.datetime64("NaT"), dtype="datetime64[ns]"
    )

    for i in range(lookback, len(highs) - lookback):
        if (i + n) >= len(date):
            continue

        is_fractal_top = True
        is_fractal_bottom = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i - j] or highs[i] <= highs[i + j]:
                is_fractal_top = False
            if lows[i] >= lows[i - j] or lows[i] >= lows[i + j]:
                is_fractal_bottom = False

        if is_fractal_top:
            fractal_top[i] = highs[i]
            # Convert to numpy datetime64 explicitly
            fractal_time_top[i] = pd.to_datetime(date[i + n]).to_datetime64()
        if is_fractal_bottom:
            fractal_bottom[i] = lows[i]
            fractal_time_bottom[i] = pd.to_datetime(date[i + n]).to_datetime64()

    df["fractal_top"] = fractal_top
    df["fractal_bottom"] = fractal_bottom
    df["fractal_time_top"] = fractal_time_top
    df["fractal_time_bottom"] = fractal_time_bottom

    df["fractal_top"] = df["fractal_top"].shift(2)
    df["fractal_bottom"] = df["fractal_bottom"].shift(2)
    df["fractal_time_top"] = df["fractal_time_top"].shift(2)
    df["fractal_time_bottom"] = df["fractal_time_bottom"].shift(2)

    df["fractal_bottom"].ffill(inplace=True)
    df["fractal_top"].ffill(inplace=True)
    df["fractal_time_top"].ffill(inplace=True)
    df["fractal_time_bottom"].ffill(inplace=True)

    return df


def fetch_historical_data(timeframe):
    try:

        now = datetime.now(tz=timezone.utc)
        rounded = now.replace(second=0, microsecond=0, minute=(now.minute // TF) * TF)
        last_complete = rounded - timedelta(minutes=TF)

        candleData = list(
            candles.find({"symbol": candleSymbol}, {"_id": 0})
            .sort("timestamp", pymongo.ASCENDING)
            .limit(40000)
        )

        if not candleData:
            logger.error("No data returned from MongoDB")
            return None

        candleDf = pd.DataFrame(candleData)

        required_columns = [
            "timestamp",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        missing_columns = [
            col for col in required_columns if col not in candleDf.columns
        ]

        if missing_columns:
            logger.error(f"Missing required columns in data: {missing_columns}")
            logger.error(f"Available columns: {candleDf.columns.tolist()}")
            return None

        candleDf = candleDf[required_columns].copy()

        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            candleDf[col] = pd.to_numeric(candleDf[col], errors="coerce")

        candleDf["timestamp"] = pd.to_numeric(candleDf["timestamp"], errors="coerce")

        candleDf["date"] = pd.to_datetime(candleDf["date"], utc=True, errors="coerce")

        candleDf.dropna(inplace=True)

        if candleDf.empty:
            logger.error("No valid data after cleaning")
            return None

        df = cand_conv(timeframe, candleDf)
        df.reset_index(inplace=True, drop=True)

        if df["date"].dt.tz is None:
            df["date"] = df["date"].dt.tz_localize("UTC")
        else:
            df["date"] = df["date"].dt.tz_convert("UTC")

        if last_complete.tzinfo is None:
            last_complete = last_complete.replace(tzinfo=timezone.utc)
        else:
            last_complete = last_complete.astimezone(timezone.utc)

        df = df[df["date"] <= last_complete]

        if df.empty:
            logger.warning("No complete candles available after filtering")
            return None

        return df

    except Exception as e:
        logger.error(f"Error in fetch_historical_data: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def cand_conv2(timeframe, df, z=False):
    if df.empty:
        return df

    last_date = df.loc[df.index[-1], "date"]
    total_minutes = last_date.hour * 60 + last_date.minute
    delete_row = False

    if isinstance(timeframe, str):
        try:
            timeframe = int(timeframe.replace("min", ""))
        except ValueError:
            logger.error(f"Invalid timeframe format: {timeframe}")
            raise

    if z:
        if total_minutes % timeframe == 0:
            df = df[:-1]
            delete_row = False
        elif (total_minutes + 1) % timeframe == 0:
            delete_row = False
        elif total_minutes + timeframe > 1440:
            delete_row = False
        else:
            delete_row = True

    ohlc_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    df = df.set_index("date")

    try:
        df = df.resample(f"{timeframe}T").apply(ohlc_dict)
    except Exception as e:
        logger.error(f"Error in resampling: {str(e)}")
        logger.error(f"timeframe value: {timeframe}, type: {type(timeframe)}")
        raise

    df = df.reset_index()

    if delete_row and z:
        df = df[:-1]

    return df


def cand_conv(timeframe, df):
    if df.empty:
        return df

    # Ensure date is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    # Sort by date to ensure proper processing
    df = df.sort_values("date").reset_index(drop=True)

    # Find all midnight timestamps
    c = list(df.loc[(df["date"].dt.hour == 0) & (df["date"].dt.minute == 0)].index)
    cd = pd.DataFrame()

    # Handle different cases based on where the day boundaries are
    if not c or (len(c) == 1 and c[0] == 0):
        x = cand_conv2(timeframe, df, z=True)
        if not x.empty:
            cd = pd.concat([cd, x], ignore_index=True)
            if not cd.empty:
                cd = cd.iloc[1:].reset_index(drop=True)
        return cd

    elif len(c) == 1 and c[0] != 0:
        x1 = cand_conv2(timeframe, df[: c[0]])
        x2 = cand_conv2(timeframe, df[c[0] :], z=True)
        cd = pd.concat([x1[1:], x2], ignore_index=True)
        return cd

    elif c[0] == 0:
        c = c[1:]

    # Process data between day boundaries
    for i in range(len(c)):
        if i == 0:
            x = cand_conv2(timeframe, df[: c[i]])
        elif i == len(c) - 1:
            x1 = cand_conv2(timeframe, df[c[i - 1] : c[i]])
            x2 = cand_conv2(timeframe, df[c[i] :], z=True)
            cd = pd.concat([cd, x1, x2], ignore_index=True)
            continue
        else:
            x = cand_conv2(timeframe, df[c[i - 1] : c[i]])

        if not x.empty:
            cd = pd.concat([cd, x], ignore_index=True)

    # Clean up the result
    if not cd.empty:
        cd = cd.iloc[1:].reset_index(drop=True)

    return cd


def SuperTrend(df, atr_period, st_factor):

    y = df[["high", "low", "close"]].copy()

    hl = pd.Series(y["high"] - y["low"]).abs()
    hc = pd.Series(y["high"] - y["close"].shift()).abs()
    cl = pd.Series(y["close"].shift() - y["low"]).abs()
    hcl = pd.concat([hl, hc, cl], axis=1)
    tr = hcl.max(axis=1)

    y["hl2"] = (y["high"] + y["low"]) / 2
    y["atr"] = (
        tr.ewm(alpha=1 / atr_period, min_periods=atr_period).mean().round(7).fillna(0)
    )
    y["UB"] = y["hl2"] + st_factor * y["atr"]
    y["LB"] = y["hl2"] - st_factor * y["atr"]

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

    df["Direction"] = x["direction"]


def analyze_market_data(df, ATR_PERIOD=14, FRACTAL_PERIOD=5, ST_FACTOR=3):
    df = df.copy()

    df["atr"] = ATR(df, ATR_PERIOD)

    df = detect_fractals(df, consecutive=FRACTAL_PERIOD)

    SuperTrend(df, ATR_PERIOD, ST_FACTOR)

    fractal_cols = [
        "fractal_top",
        "fractal_bottom",
        "fractal_time_top",
        "fractal_time_bottom",
    ]
    df[fractal_cols] = df[fractal_cols].ffill()

    return df


def check_for_entry_signals(df, trade_dict):

    open_positions = list(
        PositionCollection.find({"Strategy": STRATEGY, "Status": "Open"})
    )
    if len(open_positions) > 0:
        logger.info("Open positions exist, skipping entry signal generation")
        return None

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    prev_frac = trade_dict["PREV FRACTAL"]
    prev_exit_time = trade_dict["ExitTime"]
    logger.info(f"Previous fractal: {prev_frac}")
    logger.info(f"Previous exit time: {prev_exit_time}")

    # # 1min candle
    ticker = Ticks.find_one({"symbol": candleSymbol})
    if ticker is None:
        logger.warning("No ticker data found for candleSymbol")
        return None

    # Find the most recent valid fractal top and bottom
    recent_top = None
    recent_bottom = None

    # Look for the most recent fractal top
    for i in range(len(df) - 1, 0, -1):
        if not np.isnan(df.iloc[i]["fractal_top"]):
            recent_top = df.iloc[i]
            break

    # Look for the most recent fractal bottom
    for i in range(len(df) - 1, 0, -1):
        if not np.isnan(df.iloc[i]["fractal_bottom"]):
            recent_bottom = df.iloc[i]
            break

    # No signals if we don't have recent fractals
    if recent_top is None or recent_bottom is None:
        return None

    direction = latest.get("Direction", None)
    if direction is None:
        logger.warning(
            "SuperTrend direction not found in dataframe, cannot determine trend direction"
        )
        return None

    latest_time = pd.to_datetime(ticker["date"], utc=True)
    fractal_time_top = pd.to_datetime(recent_top["fractal_time_top"], utc=True)
    fractal_time_bottom = pd.to_datetime(recent_bottom["fractal_time_bottom"], utc=True)

    # Ensure prev_exit_time is timezone-aware
    if prev_exit_time.tzinfo is None:
        prev_exit_time = pd.to_datetime(prev_exit_time).tz_localize("UTC")

    logger.info(f"Latest candle: {latest.to_dict()}")
    logger.info(f"Previous candle: {prev.to_dict()}")
    logger.info(f"Ticker: {ticker}")
    logger.info(f"direction: {direction}")

    # Check for buy signal - price breaks above recent fractal top and SuperTrend direction is -1 (bullish)
    if (
        ticker["high"] > recent_top["fractal_top"]
        and prev_frac != recent_top["fractal_top"]
        and direction == -1
        and fractal_time_top > prev_exit_time
    ):

        logger.info(f"Latest candle: {latest.to_dict()}")
        logger.info(f"Previous candle: {prev.to_dict()}")
        logger.info(f"Ticker: {ticker}")

        entry_price = ticker["close"]
        atr_sl = entry_price - (SL_FACTOR * latest["atr"])
        take_profit = entry_price + (TP_FACTOR * latest["atr"])

        qty = 0.3  # (CAPITAL * LEVERAGE) / entry_price
        # qty = min(0.3,qty)

        stop_loss = atr_sl

        return {
            "Signal": "BUY",
            "EntryPrice": float(entry_price),
            "atr_sl": float(atr_sl),
            "StopLoss": float(stop_loss),
            "Target": float(take_profit),
            "Atr": float(latest["atr"]),
            "FractalPrice": float(recent_top["fractal_top"]),
            "FractalTime": fractal_time_top,
            "EntryTime": latest_time,
            "Direction": int(direction),
        }

    # Check for sell signal - price breaks below recent fractal bottom and SuperTrend direction is 1 (bearish)
    if (
        ticker["low"] < recent_bottom["fractal_bottom"]
        and prev_frac != recent_bottom["fractal_bottom"]
        and direction == 1
        and fractal_time_bottom > prev_exit_time
    ):

        logger.info(f"Latest candle: {latest.to_dict()}")
        logger.info(f"Previous candle: {prev.to_dict()}")
        logger.info(f"Ticker: {ticker}")

        entry_price = ticker["close"]
        atr_sl = entry_price + (SL_FACTOR * latest["atr"])
        take_profit = entry_price - (TP_FACTOR * latest["atr"])

        qty = 0.3  # (CAPITAL * LEVERAGE) / entry_price

        stop_loss = atr_sl

        return {
            "Signal": "SELL",
            "EntryPrice": float(entry_price),
            "atr_sl": float(atr_sl),
            "StopLoss": float(stop_loss),
            "Target": float(take_profit),
            "Atr": float(latest["atr"]),
            "FractalPrice": float(recent_top["fractal_bottom"]),
            "FractalTime": fractal_time_bottom,  # #FratalTime
            "EntryTime": latest_time,
            "Direction": int(direction),
        }

    return None


def execute_trade(signal):
    try:
        # Calculate position size
        qty = 0.3  # (CAPITAL * LEVERAGE) / signal['EntryPrice']
        global POSITION_ID
        if "SOL" in SYMBOL:
            qty = int(qty)

        POSITION_ID = int(time.perf_counter_ns())

        position_doc = {
            "Strategy": STRATEGY,
            "ID": POSITION_ID,
            "Symbol": SYMBOL,
            "Side": str(signal["Signal"]),
            "Condition": "Executed",
            "EntryPrice": float(signal["EntryPrice"]),
            "Qty": round(qty, 3),
            "StopLoss": round(float(signal["StopLoss"]), 2),
            "Target": round(float(signal["Target"]), 2),
            "Atr": round(float(signal["Atr"]), 2),
            "FractalPrice": round(float(signal["FractalPrice"]), 2),
            "FractalTime": pd.to_datetime(signal["FractalTime"]).to_pydatetime(),
            "EntryTime": pd.to_datetime(signal["EntryTime"]).to_pydatetime(),
            "Direction": int(signal["Direction"]),
            "Status": "Open",
            "UpdateTime": 0,
        }

        PositionCollection.insert_one(position_doc)

        entry_doc = {
            "Strategy": STRATEGY,
            "ID": POSITION_ID,
            "Symbol": SYMBOL,
            "Side": str(signal["Signal"]),
            "StopLoss": round(float(signal["StopLoss"]), 2),
            "Target": round(float(signal["Target"]), 2),
            "Price": float(signal["EntryPrice"]),
            "OrderTime": pd.to_datetime(signal["EntryTime"]).to_pydatetime(),
            "OrderType": "MARKET",
            "Qty": round(qty, 3),
            "UpdateTime": 0,
            "Users": {},
        }

        TradeCollection.insert_one(entry_doc)

        # Store position in position collection
        live_doc = {
            "Strategy": STRATEGY,
            "PREV FRACTAL": signal["FractalPrice"],
            "EntryID": POSITION_ID,
            "Symbol": SYMBOL,
            "Side": signal["Signal"],
            "EntryTime": signal["EntryTime"],
            "EntryPrice": signal["EntryPrice"],
            "Qty": round(qty, 3),
            "FractalPrice": signal["FractalPrice"],
            "FractalTime": signal["FractalTime"],
            "ATR": signal["Atr"],
            "Direction": round(signal["Direction"], 2),
            "StopLoss": round(signal["StopLoss"], 2),
            "Target": round(float(signal["Target"]), 2),
            "Status": "Open",
        }

        LiveCollection.update_one(
            {"Strategy": STRATEGY}, {"$set": live_doc}, upsert=True
        )

        return POSITION_ID

    except Exception as e:
        logger.error(f"Error executing trade: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def check_open_positions():

    try:
        # Get current price
        ticker = Ticks.find_one({"symbol": candleSymbol})
        current_price = ticker["close"]

        # Get open positions
        open_positions = list(
            PositionCollection.find({"Strategy": STRATEGY, "Status": "Open"})
        )
        # logger.info(open_positions)
        for position in open_positions:
            exit_triggered = False
            exit_type = None
            exit_price = None

            # Check if stop loss hit
            if position["Side"] == "BUY":
                if current_price <= position["StopLoss"]:
                    exit_triggered = True
                    exit_type = "StopLoss"
                    exit_price = position["StopLoss"]

                    flt = {
                        "Condition": "Executed",
                        "UpdateTime": datetime.now(tz=pytz.utc),
                    }

                    TradeCollection.update_one({"ID": position["ID"]}, {"$set": flt})
                elif current_price >= position["Target"]:
                    exit_triggered = True
                    exit_type = "Target"
                    exit_price = position["Target"]
                    flt = {
                        "Condition": "Cancel",
                        "UpdateTime": datetime.now(tz=pytz.utc),
                    }
                    TradeCollection.update_one({"ID": position["ID"]}, {"$set": flt})
            else:
                if current_price >= position["StopLoss"]:
                    exit_triggered = True
                    exit_type = "StopLoss"
                    exit_price = position["StopLoss"]
                    flt = {
                        "Condition": "Executed",
                        "UpdateTime": datetime.now(tz=pytz.utc),
                    }
                    TradeCollection.update_one({"ID": position["ID"]}, {"$set": flt})

                elif current_price <= position["Target"]:
                    exit_triggered = True
                    exit_type = "Target"
                    exit_price = position["Target"]
                    flt = {
                        "Condition": "Cancel",
                        "UpdateTime": datetime.now(tz=pytz.utc),
                    }
                    TradeCollection.update_one({"ID": position["ID"]}, {"$set": flt})

            if exit_triggered:
                # Calculate PNL
                if position["Side"] == "BUY":
                    pnl = (exit_price - position["EntryPrice"]) * position["Qty"]
                else:
                    pnl = (position["EntryPrice"] - exit_price) * position["Qty"]

                # Update position status
                PositionCollection.update_one(
                    {"Strategy": STRATEGY, "ID": position["ID"]},
                    {
                        "$set": {
                            "Status": "Closed",
                            "ExitPrice": exit_price,
                            "ExitTime": ticker["date"],
                            "ExitType": exit_type,
                            "PNL": pnl,
                        }
                    },
                )

                # # Create trade record for history
                # trade_record = {
                #     "Strategy": STRATEGY,
                #     "ID": position['ID'],
                #     "Symbol": position['Symbol'],
                #     "Side": position['Side'],
                #     "EntryPrice": position['EntryPrice'],
                #     "ExitPrice": exit_price,
                #     "Qty": position['Qty'],
                #     "EntryTime": position['EntryTime'],
                #     "ExitTime": ticker['date'],
                #     "ExitType": exit_type,
                #     "PNL": pnl,
                #     "Status": "Closed",
                #     "StopLoss": position.get('StopLoss'),
                #     "Target": position.get('Target'),
                #     "Atr": position.get('Atr'),
                #     "FractalPrice": position.get('FractalPrice'),
                #     "FractalTime": position.get('FractalTime'),
                #     "Direction": position.get('Direction')
                # }

                # Update live collection
                LiveCollection.update_one(
                    {"Strategy": STRATEGY},
                    {
                        "$set": {
                            "ExitPrice": exit_price,
                            "ExitTime": ticker["date"],
                            "ExitType": exit_type,
                            "PNL": pnl,
                            "Status": "Completed",
                        }
                    },
                )

                # Insert trade record into history
                # TradeCollection.insert_one(trade_record)

                logger.info(f"Position closed: {position}")

    except Exception as e:
        logger.error(f"Error checking positions: {str(e)}")
        logger.error(traceback.format_exc())

    except Exception as e:
        logger.error(f"Error updating OHLC database: {str(e)}")
        logger.error(traceback.format_exc())


def main():
    logger.info(f"Starting {STRATEGY}")

    while True:
        tm = [x for x in range(0, 60) if x % TF == 0]
        dt_m = int(datetime.now().minute)
        if dt_m in tm:
            logger.info(dt_m)
            try:
                df = fetch_historical_data(TIMEFRAME)

                if df is None:
                    logger.warning(
                        "No data returned from fetch_historical_data, waiting 60 seconds..."
                    )
                    time.sleep(60)
                    continue

                if len(df) < 5:  # Need enough data for analysis
                    logger.warning(
                        f"Not enough data points for analysis ({len(df)} < 5), waiting..."
                    )
                    time.sleep(60)
                    continue

                df = analyze_market_data(df)

                open_positions = list(
                    PositionCollection.find({"Strategy": STRATEGY, "Status": "Open"})
                )

                if len(open_positions) == 0:
                    trade_dict = liveUpdate(STRATEGY)
                    # Check for entry signals
                    signal = check_for_entry_signals(df, trade_dict)

                    # Execute trade if signal found
                    if signal is not None:
                        trade_id = execute_trade(signal)
                        if trade_id:
                            logger.info(f"Trade executed with ID: {trade_id}")

                    time.sleep(60)

            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                logger.error(traceback.format_exc())
                time.sleep(60)

        check_open_positions()


if __name__ == "__main__":
    # MongoDB connection
    load_dotenv()

    MONGO_LINK = os.getenv("MONGO_URL")
    MONGO_DEV_DB_NAME = os.getenv("MONGO_DB_NAME")

    print(MONGO_LINK)
    print(MONGO_DEV_DB_NAME)

    # STRATEGY = "ETH Multiplier"
    STRATEGY = str(argv[1])
    STRATEGY = STRATEGY.replace("_", " ").strip()

    if "BTC" in STRATEGY:
        SYMBOL = "BTC-USDT"
        candleSymbol = "BTCUSDT"
    elif "ETH" in STRATEGY:
        SYMBOL = "ETH-USDT"
        candleSymbol = "ETHUSDT"
    elif "SOL" in STRATEGY:
        SYMBOL = "SOL-USDT"
        candleSymbol = "SOLUSDT"
    else:
        print("Symbol not allow")
        # sys.exit(1)

    # Setup logger
    current_file = str(os.path.basename(__file__)).replace(".py", "")
    folder = file_path_locator()
    logs_dir = path.join(path.normpath(folder), "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    LOG_file = f"{logs_dir}/{STRATEGY}.log"

    logger = setup_logger(
        name=current_file, log_to_file=True, log_file=LOG_file, capture_print=False
    )

    TF = 3
    TIMEFRAME = f"{TF}min"
    FRACTAL_PERIOD = 5
    ATR_PERIOD = 14
    ST_FACTOR = 3.0
    SL_FACTOR = 1.0
    TP_FACTOR = 2.0
    LEVERAGE = 20
    CAPITAL = 150

    # Initialize MongoDB connections
    myclient = pymongo.MongoClient(MONGO_LINK)
    db_name = MONGO_DEV_DB_NAME
    mydb = myclient[db_name]
    LiveCollection = mydb["live"]
    PositionCollection = mydb["position"]
    TradeCollection = mydb["trades"]
    candles = mydb["candleData"]
    Ticks = mydb["ticks"]

    trade_dict = liveUpdate(STRATEGY)

    try:
        import pause

        dt = datetime.now()
        minutes = dt.minute
        tt = minutes + 1
        if tt > 59:
            tt = 0
            hours = dt.hour + 1
            pause.until(dt.replace(hour=hours, minute=tt, second=0))

        else:
            pause.until(dt.replace(minute=tt, second=0))

        main()

    except KeyboardInterrupt:
        logger.info("Strategy stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
