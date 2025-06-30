from datetime import datetime, timedelta, timezone
import ccxt
import pandas as pd
import pdb

begin_time = datetime.now(timezone.utc)

API_KEY = "v8MSBjTa6pgtvV3SphSMBtUfWjArz5ZlxO8Wi9wrqokZ3i4DjbC0Hy8sZ7wJ6tHH"
API_SECRET = "y3IPkb1bZqIFFsOLBXjGAlNDdsFkBhtC8botTXvue9k8la0ewCP8mrYVFFhp3mcv"

binance = ccxt.binancecoinm(

    {
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {
            "defaultType": "delivery",  # // spot, future, margin
        },
    }
)

try:
    dfx = pd.read_csv("ethusd_merged.csv", parse_dates=["date"])
    start_date = dfx.iloc[-1]["date"] + timedelta(minutes=1)
    add = 1
except:
    start_date = datetime.now(timezone.utc) - timedelta(days=381)
    real_start_date = int(start_date.replace().timestamp() * 1000)
    add = 0

real_start_date = int(start_date.replace().timestamp() * 1000)
dfx = pd.DataFrame()

while True:

    try:

        data = binance.fetch_ohlcv(
            "ETH/USD", timeframe="1m", since=real_start_date, limit=500
        )
        df = pd.DataFrame(data)
        real_start_date = int(df.loc[df.index[-1], 0]) + 60000

        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df[["date", "open", "high", "low", "close", "volume"]]

        dfx = pd.concat([dfx, df], ignore_index=True)

        if (
            real_start_date
            > int(datetime.now(timezone.utc).replace().timestamp()) * 1000
        ):
            break

    except Exception as e:

        real_start_date = int(
            start_date.replace().timestamp() * 1000
        )
        dfx = pd.DataFrame()
        print(datetime.now(timezone.utc))
        print(e)
        continue

if add:
    dfx[:-1].to_csv("ethusd_merged.csv", mode="a", header=False, index=False)
    print("File Updated", datetime.now(timezone.utc))
    print(datetime.now(timezone.utc) - begin_time)
    # pdb.set_trace()
else:
    dfx[:-1].to_csv("ethusd_merged.csv", index=False)
    print("New File Created", datetime.now(timezone.utc))
    print(datetime.now(timezone.utc) - begin_time)
