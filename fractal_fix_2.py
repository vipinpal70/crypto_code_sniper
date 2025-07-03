import pandas as pd 
import numpy as np
import pymongo
from datetime import datetime


link = "mongodb+srv://btcusdcoin:5%40cWKK32u6xCDMg@btccoin.7sujtcg.mongodb.net/admin?authSource=admin&replicaSet=atlas-10fy2v-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true"

client = pymongo.MongoClient(link)


candles = client.Candles['BTC/USD']

df = pd.DataFrame(list(candles.find()))

df['date'] = pd.to_datetime(df['date'])
flt = df[(df['date'] > datetime(2025,5,5)) & (df['date'] <= datetime(2025,5,25,2,10,0,0))]
flt = flt[["date","open","high","low","close","volume"]]

def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if not all(col in df.columns for col in ['date', 'open', 'high', 'low', 'close', 'volume']):
        raise ValueError("DataFrame must contain 'date', 'open', 'high', 'low', 'close', 'volume' columns.")
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    original_last_timestamp = df['date'].max()

    df = df.set_index('date').sort_index()

    ohlcv_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    resampled_df = df.resample(timeframe, origin='start_day').agg(ohlcv_dict)

    resampled_df.dropna(subset=['close'], inplace=True)

    timeframe_delta = pd.to_timedelta(timeframe)

    resampled_df = resampled_df[resampled_df.index + timeframe_delta - pd.Timedelta(nanoseconds=1) <= original_last_timestamp]

    return resampled_df.reset_index()


def detect_fractals(df: pd.DataFrame, lookback_periods: int = 2) -> pd.DataFrame:
    
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    if lookback_periods < 1:
        raise ValueError("lookback_periods must be at least 1.")

    df_result = df.sort_values(by='date', ascending=True).copy()

    if df_result.empty:
        return pd.DataFrame(columns=[
            'date', 'open', 'high', 'low', 'close', 'volume',
            'fractal_top', 'fractal_bottom', 'fractal_time_top', 'fractal_time_bottom'
        ])


    highs = df_result['high'].values
    lows = df_result['low'].values
    dates = df_result['date'].values
    
    assert isinstance(len(highs), int), f"len(highs) is not an integer! It's {type(len(highs))}"


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
    df_result['_temp_detected_fractal_top_price'] = detected_fractal_top_price
    df_result['_temp_detected_fractal_bottom_price'] = detected_fractal_bottom_price
    df_result['_temp_detected_fractal_top_time'] = detected_fractal_top_time
    df_result['_temp_detected_fractal_bottom_time'] = detected_fractal_bottom_time


    df_result['fractal_top'] = df_result['_temp_detected_fractal_top_price'].ffill()
    df_result['fractal_bottom'] = df_result['_temp_detected_fractal_bottom_price'].ffill()
    df_result['fractal_time_top'] = df_result['_temp_detected_fractal_top_time'].ffill()
    df_result['fractal_time_bottom'] = df_result['_temp_detected_fractal_bottom_time'].ffill()
    
    df_result['fractal_time_top'] = pd.to_datetime(df_result['fractal_time_top'])
    df_result['fractal_time_bottom'] = pd.to_datetime(df_result['fractal_time_bottom'])
    
    df_result['fractal_top'] = df_result['fractal_top'].shift(2)
    df_result['fractal_bottom'] = df_result['fractal_bottom'].shift(2)
    df_result['fractal_time_top'] = df_result['fractal_time_top'].shift(2)
    df_result['fractal_time_bottom'] = df_result['fractal_time_bottom'].shift(2)
    


    # Drop temporary columns
    df_result = df_result.drop(columns=[
        '_temp_detected_fractal_top_price', '_temp_detected_fractal_bottom_price',
        '_temp_detected_fractal_top_time', '_temp_detected_fractal_bottom_time'
    ])

    return df_result



ress = resample(flt,"15min")

test = detect_fractals(ress)

print(test.tail())


