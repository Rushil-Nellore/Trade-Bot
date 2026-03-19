import requests
import pandas as pd
from datetime import datetime

def get_price_data(symbol="BTCUSDT", interval="1h", limit=100):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,    # BTC/USD pair
        "interval": interval, # 1h = one candle per hour
        "limit": limit        # how many candles to fetch
    }
    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    # keep only what we need
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["close"] = pd.to_numeric(df["close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    return df

df = get_price_data()
print(df.tail())
