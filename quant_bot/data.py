from __future__ import annotations

import time

import pandas as pd
import requests

from quant_bot.config import BINANCE_KLINES_URL, REQUEST_TIMEOUT


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MA_10"] = df["close"].rolling(window=10).mean()
    df["MA_30"] = df["close"].rolling(window=30).mean()
    df["MA_200"] = df["close"].rolling(window=200).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def get_price_data(symbol: str, interval: str, days: int) -> pd.DataFrame:
    all_data = []
    end_time = int(time.time() * 1000)
    hours_needed = days * 24
    candles_fetched = 0

    while candles_fetched < hours_needed:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": 1000,
            "endTime": end_time,
        }
        response = requests.get(BINANCE_KLINES_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if not data:
            break

        all_data = data + all_data
        end_time = data[0][0] - 1
        candles_fetched += len(data)

    if not all_data:
        raise RuntimeError("No candle data returned from Binance.")

    df = pd.DataFrame(
        all_data,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return add_indicators(df)

