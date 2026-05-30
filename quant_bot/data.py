from __future__ import annotations  # enables newer type-hint syntax (e.g. list[str]) on Python 3.9 and earlier without errors

import datetime  # Python's standard date/time library — used to convert a "YYYY-MM-DD" string into a Unix timestamp
import logging  # Python's built-in logging library — lets us print timestamped messages with severity levels instead of plain print()
import time  # standard library module for time-related functions — used here to get the current Unix timestamp in seconds

import pandas as pd  # pandas: the main Python data-table library — pd.DataFrame is a 2D table, pd.to_datetime converts raw numbers to dates
import requests  # third-party HTTP library — makes it easy to send web requests (GET, POST) to APIs like Binance
from requests.adapters import HTTPAdapter  # HTTPAdapter: attaches custom connection behaviour (like retry logic) to a requests.Session
from urllib3.util.retry import Retry  # Retry: urllib3 class that defines how many times to retry and which HTTP error codes should trigger a retry

from quant_bot.config import BINANCE_KLINES_URL, REQUEST_TIMEOUT  # import only the two constants this file actually uses from config.py

logger = logging.getLogger(__name__)  # getLogger(__name__): create a logger whose name is the module path ("quant_bot.data") — log lines will show which file produced them

# ── BUG FIX: interval-aware candle count ──────────────────────────────────────
# The original code used hours_needed = days * 24, which only worked for 1h candles.
# For a 4h interval, that would try to fetch 4x too many batches; for 1d it would try
# 24x too many. This lookup table maps each interval to the correct candles-per-day so
# the while loop stops at the right count for any interval.
_CANDLES_PER_DAY: dict[str, int] = {  # maps interval string to how many candles fit in exactly one calendar day
    "1m":  1440,  # 60 minutes × 24 hours = 1440 one-minute candles per day
    "5m":   288,  # 12 per hour × 24 hours = 288
    "15m":   96,  # 4 per hour × 24 hours = 96
    "30m":   48,  # 2 per hour × 24 hours = 48
    "1h":    24,  # 1 per hour × 24 hours = 24
    "4h":     6,  # 1 per 4 hours × 24 hours = 6
    "1d":     1,  # 1 per day = 1
}


def _make_session() -> requests.Session:  # private helper (leading underscore = internal use only) — returns a Session that automatically retries failed requests
    session = requests.Session()  # Session(): creates a persistent HTTP connection object that reuses headers/cookies and is faster than making a fresh connection each time
    retry = Retry(  # Retry(): configure the automatic retry policy
        total=3,  # retry up to 3 extra times before giving up on a request
        backoff_factor=0.5,  # wait 0.5 s before retry 1, 1.0 s before retry 2, 2.0 s before retry 3 — exponential back-off reduces load on a struggling server
        status_forcelist=[429, 500, 502, 503, 504],  # automatically retry if Binance responds with these codes: 429 = rate-limited, 5xx = server-side errors
        allowed_methods=["GET"],  # only retry GET requests — safe to repeat; POST/PUT requests could create duplicate orders if retried
        raise_on_status=False,  # do not raise an exception inside the retry logic itself — we call raise_for_status() manually later
    )
    adapter = HTTPAdapter(max_retries=retry)  # HTTPAdapter(max_retries=retry): wrap the Retry config inside an adapter that the session will use for every outgoing request
    session.mount("https://", adapter)  # mount(): attach this adapter to all URLs beginning with "https://" — Binance uses HTTPS
    session.mount("http://", adapter)  # also attach to "http://" as a fallback in case the connection is redirected to plain HTTP
    return session  # hand the configured session back to the caller


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:  # takes a raw OHLCV table and returns it with moving-average and RSI columns appended
    df = df.copy()  # copy(): make an independent copy so we do not accidentally modify the DataFrame the caller passed in — avoids surprising side-effects
    df["MA_10"] = df["close"].rolling(window=10).mean()  # rolling(window=10): slide a 10-row window along the close column; .mean(): average the window's values → 10-period moving average; first 9 rows are NaN (not enough history yet)
    df["MA_30"] = df["close"].rolling(window=30).mean()  # same logic, 30-period window — reacts more slowly to short-term price swings; first 29 rows are NaN
    df["MA_200"] = df["close"].rolling(window=200).mean()  # 200-period MA — represents the long-term price trend; first 199 rows are NaN

    delta = df["close"].diff()  # diff(): subtract each row's close price from the next row's — positive value = price went up, negative = price went down
    gain = delta.clip(lower=0)  # clip(lower=0): set all negative values to 0 — keeps only the upward price moves; downward moves become 0
    loss = -delta.clip(upper=0)  # clip(upper=0): set all positive values to 0, keeping only negatives; negate the result so losses are expressed as positive numbers
    avg_gain = gain.rolling(window=14).mean()  # rolling 14-period average of gains — one of the two inputs to RSI
    avg_loss = loss.rolling(window=14).mean()  # rolling 14-period average of losses — the other input to RSI
    rs = avg_gain / avg_loss.replace(0, float("nan"))  # RS = Relative Strength = avg_gain / avg_loss; .replace(0, NaN): use NumPy NaN (not pd.NA) so the resulting RSI column stays a plain float64 that matplotlib can plot
    df["RSI"] = (100 - (100 / (1 + rs))).astype(float)  # standard RSI formula — converts RS to a 0-100 scale; .astype(float): force a plain float dtype so matplotlib never sees pd.NA values (which it can't handle)

    # ── MACD (Moving Average Convergence Divergence) ──────────────────────────
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()  # ewm(span=12): exponentially weighted moving average — gives more weight to recent prices; adjust=False uses the standard MACD formula
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()  # the slower EMA — span=26 means more lag
    df["MACD"] = ema_12 - ema_26  # MACD line = fast EMA minus slow EMA — positive when momentum is up, negative when down
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()  # signal line = 9-period EMA of MACD — used to smooth the MACD line for crossover signals

    # ── Bollinger Bands (volatility envelope) ─────────────────────────────────
    bb_mid = df["close"].rolling(window=20).mean()  # 20-period MA — the middle of the band
    bb_std = df["close"].rolling(window=20).std()  # rolling 20-period standard deviation — measures price volatility
    df["BB_upper"] = bb_mid + 2 * bb_std  # upper band: 2 standard deviations above the mean
    df["BB_lower"] = bb_mid - 2 * bb_std  # lower band: 2 standard deviations below the mean

    # ── ATR (Average True Range — measures volatility in dollar terms) ────────
    high_low = df["high"] - df["low"]  # candle range: high minus low
    high_close = (df["high"] - df["close"].shift()).abs()  # .shift(): previous candle's close; .abs(): absolute value — gap from previous close to current high
    low_close = (df["low"] - df["close"].shift()).abs()  # gap from previous close to current low
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)  # pd.concat(axis=1): stack the 3 series as columns; .max(axis=1): take the largest of the 3 per row → True Range
    df["ATR"] = true_range.rolling(window=14).mean().astype(float)  # 14-period MA of True Range = ATR; .astype(float): keep plain float dtype

    return df  # return the DataFrame now enriched with MA, RSI, MACD, Bollinger Bands, and ATR columns


def get_price_data(
    symbol: str,  # trading pair e.g. "BTCUSDT"
    interval: str,  # candle width e.g. "1h"
    days: int,  # how many days of history to fetch
    end_date: str | None = None,  # optional end date as "YYYY-MM-DD" string — if None, fetches up to today; set this to test historical periods e.g. "2021-11-10" for the 2021 bull run peak
) -> pd.DataFrame:  # returns a DataFrame with OHLCV columns plus MA_10, MA_30, MA_200, RSI
    """Fetch OHLCV candles from Binance and return a DataFrame with indicators.

    Fetches data in backward-walking 1000-candle batches until ``days`` worth
    of history is collected.  Uses a session with automatic retries so transient
    network blips do not immediately crash the pipeline.

    Pass ``end_date="YYYY-MM-DD"`` to fetch a historical window instead of the
    most recent data.  For example, ``end_date="2021-11-10", days=365`` fetches
    the 365 days leading up to the 2021 BTC all-time high.
    """
    session = _make_session()  # build the retry-enabled HTTP session using the private helper above
    all_data: list = []  # empty list that will accumulate raw candle lists across all API batches

    if end_date is not None:  # if the user supplied a specific end date string
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")  # strptime(): parse "YYYY-MM-DD" string into a datetime object; "%Y-%m-%d" is the format mask matching year-month-day
        end_time = int(end_dt.timestamp() * 1000)  # .timestamp(): convert datetime to Unix seconds; ×1000: convert to milliseconds for Binance; int(): drop decimal
        logger.info("Using historical end date: %s", end_date)  # log so the user knows they're looking at historical data
    else:
        end_time = int(time.time() * 1000)  # time.time(): current time in seconds since 1970 (Unix epoch); ×1000: convert to milliseconds because Binance timestamps are in ms; int(): drop the decimal
    candles_per_day = _CANDLES_PER_DAY.get(interval, 24)  # dict.get(key, default): look up how many candles fit in a day for this interval string; if the interval is not in the table, default to 24 (hourly assumption)
    candles_needed = days * candles_per_day  # total candles to collect — correctly adapts to the candle width (BUG FIX: was always days*24 which only worked for 1h)
    candles_fetched = 0  # running counter of how many candles have been collected so far

    while candles_fetched < candles_needed:  # keep looping and fetching batches until we have enough candles
        params = {  # build the query-string dictionary for the Binance API — requests will append these as ?symbol=...&interval=...
            "symbol": symbol,    # trading pair, e.g. "BTCUSDT"
            "interval": interval,  # candle width, e.g. "1h"
            "limit": 1000,       # max candles per response — 1000 is Binance's hard limit per request
            "endTime": end_time,  # fetch candles ending at or before this millisecond timestamp — we walk backwards in time
        }
        response = session.get(BINANCE_KLINES_URL, params=params, timeout=REQUEST_TIMEOUT)  # session.get(): send the HTTP GET request; params= auto-appends the dict to the URL; timeout=: give up after REQUEST_TIMEOUT seconds and trigger a retry
        response.raise_for_status()  # raise_for_status(): throw an HTTPError exception if the response status is 4xx or 5xx — stops us from silently processing error pages
        data = response.json()  # json(): parse the HTTP response body from a JSON text string into a Python list of lists (each inner list = one candle with 12 values)
        if not data:  # if Binance returned an empty list — happens when we've requested further back than when BTC trading began
            logger.warning("Binance returned an empty batch; stopping early.")  # log a warning so the user knows we stopped before reaching the target candle count
            break  # exit the while loop immediately — no point asking for more

        all_data = data + all_data  # prepend this batch to the front of our accumulated list — Binance returns newest-first, so each older batch must go before the newer ones
        end_time = data[0][0] - 1  # data[0]: the first (oldest) candle in this batch; [0]: its opening timestamp in ms; -1: move the next request's end one ms earlier so we don't re-fetch this candle
        candles_fetched += len(data)  # add the number of candles in this batch to the running total
        logger.debug("Fetched %d candles so far (need %d).", candles_fetched, candles_needed)  # debug-level log — very detailed; only shows if log level is set to DEBUG

    if not all_data:  # if we completed all loops but collected nothing at all (e.g. wrong symbol was typed)
        raise RuntimeError("No candle data returned from Binance.")  # RuntimeError: a standard Python exception for unexpected runtime failures; the message explains what went wrong

    df = pd.DataFrame(  # pd.DataFrame(): convert the list of raw candle lists into a proper 2D table with named columns
        all_data,  # the raw data — each element is a 12-item list from Binance
        columns=[  # give each of the 12 positions a descriptive name
            "timestamp",       # candle open time in milliseconds since 1970
            "open",            # price at the very start of the candle
            "high",            # highest price reached during the candle
            "low",             # lowest price reached during the candle
            "close",           # price at the very end of the candle
            "volume",          # total BTC traded during the candle
            "close_time",      # candle close time in ms — not used
            "quote_volume",    # total USDT value traded — not used
            "trades",          # number of individual trades — not used
            "taker_buy_base",  # BTC bought by aggressive buyers — not used
            "taker_buy_quote", # USDT spent by aggressive buyers — not used
            "ignore",          # unused field Binance always includes
        ],
    )
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]  # select only the 6 columns we need — drop the other 6 to save memory and keep the table clean

    for column in ["open", "high", "low", "close", "volume"]:  # loop over the 5 numeric column names
        df[column] = pd.to_numeric(df[column])  # pd.to_numeric(): convert the column from text strings (Binance sends every value as a string) to floating-point numbers so we can do arithmetic on them

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")  # pd.to_datetime(): convert raw millisecond integers (e.g. 1716163200000) into human-readable Timestamp objects (e.g. 2025-05-20 00:00:00); unit="ms" tells pandas the input unit
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)  # drop_duplicates("timestamp"): remove any rows sharing the same timestamp (can occur at batch boundaries); sort_values: put oldest candle first; reset_index(drop=True): renumber rows 0,1,2,... cleanly after reordering

    logger.info(  # logger.info(): print an informational log line — visible at INFO level and above
        "Loaded %d candles for %s (%s) from %s to %s.",  # format string with 5 %s/%d placeholders
        len(df),  # total rows in the final DataFrame
        symbol,  # e.g. "BTCUSDT"
        interval,  # e.g. "1h"
        df["timestamp"].iloc[0].date(),  # .iloc[0]: first row; .date(): strip the time component, keep just the date
        df["timestamp"].iloc[-1].date(),  # .iloc[-1]: last row (most recent candle)
    )
    return add_indicators(df)  # pipe the clean DataFrame through add_indicators() to append MA_10, MA_30, MA_200, and RSI, then return the fully enriched table
