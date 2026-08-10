from __future__ import annotations

import datetime
import logging
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from quant_bot.config import BINANCE_KLINES_URL, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

_CANDLES_PER_DAY: dict[str, int] = {
    "1m":  1440,
    "5m":   288,
    "15m":   96,
    "30m":   48,
    "1h":    24,
    "4h":     6,
    "1d":     1,
}


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


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
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    df["RSI"] = (100 - (100 / (1 + rs))).astype(float)

    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    bb_mid = df["close"].rolling(window=20).mean()
    bb_std = df["close"].rolling(window=20).std()
    df["BB_upper"] = bb_mid + 2 * bb_std
    df["BB_lower"] = bb_mid - 2 * bb_std

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(window=14).mean().astype(float)

    df["MA_50"]  = df["close"].rolling(window=50).mean()
    df["MA_100"] = df["close"].rolling(window=100).mean()

    df["VOL_MA_20"]  = df["volume"].rolling(window=20).mean()
    df["VOL_STD_20"] = df["volume"].rolling(window=20).std()

    df["VOL_24"] = df["close"].pct_change().rolling(window=24).std().astype(float)

    df["HIGH_20"] = df["high"].rolling(window=20).max()
    df["LOW_20"]  = df["low"].rolling(window=20).min()

    return df


def get_price_data(
    symbol: str,
    interval: str,
    days: int,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV candles from Binance and return a DataFrame with indicators.

    Fetches data in backward-walking 1000-candle batches until ``days`` worth
    of history is collected.  Uses a session with automatic retries so transient
    network blips do not immediately crash the pipeline.

    Pass ``end_date="YYYY-MM-DD"`` to fetch a historical window instead of the
    most recent data.  For example, ``end_date="2021-11-10", days=365`` fetches
    the 365 days leading up to the 2021 BTC all-time high.
    """
    session = _make_session()
    all_data: list = []

    if end_date is not None:
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
        end_time = int(end_dt.timestamp() * 1000)
        logger.info("Using historical end date: %s", end_date)
    else:
        end_time = int(time.time() * 1000)
    candles_per_day = _CANDLES_PER_DAY.get(interval, 24)
    candles_needed = days * candles_per_day
    candles_fetched = 0

    while candles_fetched < candles_needed:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": 1000,
            "endTime": end_time,
        }
        response = session.get(BINANCE_KLINES_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if not data:
            logger.warning("Binance returned an empty batch; stopping early.")
            break

        all_data = data + all_data
        end_time = data[0][0] - 1
        candles_fetched += len(data)
        logger.debug("Fetched %d candles so far (need %d).", candles_fetched, candles_needed)

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

    logger.info(
        "Loaded %d candles for %s (%s) from %s to %s.",
        len(df),
        symbol,
        interval,
        df["timestamp"].iloc[0].date(),
        df["timestamp"].iloc[-1].date(),
    )
    return add_indicators(df)
