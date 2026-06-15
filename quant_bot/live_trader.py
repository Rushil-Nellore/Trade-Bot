"""Live (paper or real) trading loop — one iteration = one decision.

How it works:
    1. Fetch the most recent candles from Binance.
    2. Compute indicators and the same entry/exit rules used by the backtest.
    3. Apply the trained ML model as an entry gate (just like simulate()).
    4. If a signal fires, place a MARKET order via BinanceClient.
    5. Persist state (cash, position, peak price) to a JSON file so the next
       run picks up exactly where this one left off.

Schedule this with cron / Task Scheduler to run hourly:
    python live.py        — one iteration
    python live.py loop   — keep checking every hour
"""
from __future__ import annotations  # newer type-hint syntax on older Python

import json     # standard library — read/write the live_state.json persistence file
import logging  # for structured timestamped log output
from dataclasses import dataclass, field, asdict  # dataclass: auto-generates init; asdict: convert dataclass instance to plain dict (for JSON serialisation)
from pathlib import Path  # object-oriented filesystem paths — cleaner than string manipulation

import pandas as pd  # data-table library — used to convert kline JSON into a DataFrame

from quant_bot.binance_client import BinanceClient  # the REST client we wrote above
from quant_bot.config import (  # pull every constant the live trader needs
    BULL_POSITION_SIZE_PCT,
    BULL_REGIME_MA200_BUFFER,
    BULL_TRAILING_STOP_PCT,
    DEFAULT_INTERVAL,
    DEFAULT_SYMBOL,
    FEE_RATE,
    HARD_STOP_PCT,
    INITIAL_BALANCE,
    LIVE_STATE_PATH,
    ML_CONFIDENCE_THRESHOLD,
    POSITION_SIZE_PCT,
    TRAILING_STOP_PCT,
)
from quant_bot.data import add_indicators  # add MA/RSI/MACD/Bollinger/ATR columns to a DataFrame
from quant_bot.ml import MLModel, compute_features_at_row  # the trained model class + per-row feature builder
from quant_bot.strategy import should_exit_trade  # the same exit logic used by the backtest

logger = logging.getLogger(__name__)  # logger named "quant_bot.live_trader"


# ── Persisted state ───────────────────────────────────────────────────────────

@dataclass  # auto-generate __init__ and __repr__
class LiveState:  # everything we need to remember between runs — saved as JSON
    cash_balance: float = INITIAL_BALANCE  # how much USDT the bot thinks it has (synced from Binance on each run for safety)
    in_trade: bool = False                 # are we currently holding a position?
    btc_position: float = 0.0              # how much BTC we hold while in_trade
    buy_price: float = 0.0                 # entry price — for hard-stop calculation
    peak_price: float = 0.0                # highest price seen since entry — for trailing-stop
    entry_timestamp: str = ""              # when we entered (ISO format string for JSON-safety)
    current_trailing_stop: float = TRAILING_STOP_PCT  # per-trade trailing stop set at entry (5% normal, 10% strong uptrend)
    trade_history: list[dict] = field(default_factory=list)  # list of completed trades — same schema as backtest's trade_log


def load_state(path: str = LIVE_STATE_PATH) -> LiveState:  # read state from JSON file; return defaults if the file does not exist yet
    p = Path(path)  # Path: object-oriented path
    if not p.exists():  # first run — no state file yet
        logger.info("No state file at %s, starting fresh.", path)
        return LiveState()  # all defaults
    with p.open("r", encoding="utf-8") as f:  # open file for reading; with-statement ensures the file is closed afterwards
        data = json.load(f)  # json.load(): parse the JSON file into a Python dict
    logger.info("Loaded state: cash=$%.2f in_trade=%s", data.get("cash_balance", 0), data.get("in_trade", False))
    return LiveState(**data)  # **data: unpack the dict as keyword arguments into the dataclass constructor


def save_state(state: LiveState, path: str = LIVE_STATE_PATH) -> None:  # serialise state to JSON file
    with Path(path).open("w", encoding="utf-8") as f:  # open for writing — overwrites previous file
        json.dump(asdict(state), f, indent=2, default=str)  # json.dump(): write dict as JSON; indent=2: pretty-print; default=str: convert non-serialisable types (Timestamps) to string


# ── One iteration of the trading loop ────────────────────────────────────────

def trade_once(  # do exactly ONE decision check — designed to be called by cron/Task Scheduler each hour
    client: BinanceClient,  # the (testnet or live) Binance REST client
    ml_model: MLModel,      # the trained XGBoost model from run_ml_experiment()
    *,                      # everything after * must be passed by name — prevents argument-order mistakes
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    state_path: str = LIVE_STATE_PATH,
) -> LiveState:
    state = load_state(state_path)  # load whatever the previous run wrote — first run uses defaults

    # ── 1. Fetch enough recent candles to compute all 29 features ────────────
    #
    # We need at least 200 candles for MA200, and a few extra for safety —
    # 300 hourly candles ≈ 12.5 days of context.
    candles = client.get_klines(symbol=symbol, interval=interval, limit=300)  # Binance returns up to 1000; we use 300 — plenty for indicators

    df = pd.DataFrame(candles, columns=[  # convert raw list-of-lists into a DataFrame with named columns (same as data.py)
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]  # drop the unused columns
    for col in ["open", "high", "low", "close", "volume"]:  # convert string values to floats
        df[col] = pd.to_numeric(df[col])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")  # convert ms-since-epoch into datetime objects
    df = add_indicators(df).reset_index(drop=True)  # add MA/RSI/MACD/BB/ATR/etc — exactly like the backtest pipeline does

    # ── 2. Use the LAST FULLY-CLOSED candle as the decision point ────────────
    # The last row of the kline response is the in-progress candle — its values
    # change every second. Acting on it would mean different decisions every
    # call. Use the second-to-last row which is finalised.
    i = len(df) - 2  # index of the most recent closed candle
    if i < 200:  # safety check — should never trigger with limit=300
        logger.warning("Not enough candles loaded (%d) — skipping iteration.", len(df))
        return state

    timestamp = df["timestamp"].iloc[i]
    price = float(df["close"].iloc[i])
    ma10  = float(df["MA_10"].iloc[i])
    ma30  = float(df["MA_30"].iloc[i])
    ma200_now  = float(df["MA_200"].iloc[i])
    ma200_prev = float(df["MA_200"].iloc[i - 10])  # MA200 ten candles ago — for "is the trend rising" check
    prev_ma10  = float(df["MA_10"].iloc[i - 1])
    prev_ma30  = float(df["MA_30"].iloc[i - 1])
    rsi = float(df["RSI"].iloc[i])

    logger.info("Decision tick @ %s | price=$%.2f | in_trade=%s | cash=$%.2f", timestamp, price, state.in_trade, state.cash_balance)

    # ── 3. If we ARE in a trade, check exit conditions first ─────────────────
    if state.in_trade:
        state.peak_price = max(state.peak_price, price)  # update trailing peak
        exit_reason = should_exit_trade(
            price=price,
            buy_price=state.buy_price,
            peak_price=state.peak_price,
            trailing_stop_pct=state.current_trailing_stop,
            hard_stop_pct=HARD_STOP_PCT,
        )
        if exit_reason is not None:
            # ── Place SELL market order on Binance ───────────────────────────
            logger.info("Exit triggered (%s) at $%.2f — placing SELL order.", exit_reason, price)
            order = client.place_market_order(symbol=symbol, side="SELL", quantity=state.btc_position)
            # executedQty: actual BTC sold (may differ slightly from requested due to lot size)
            executed_qty = float(order.get("executedQty", state.btc_position))
            # cummulativeQuoteQty: total USDT received (more reliable than computing manually)
            received_usdt = float(order.get("cummulativeQuoteQty", executed_qty * price))
            profit = received_usdt - (state.btc_position * state.buy_price)  # rough P&L (ignores fees the exchange already deducted)

            state.trade_history.append({  # record completed trade for review
                "entry_time": state.entry_timestamp,
                "exit_time":  str(timestamp),
                "exit_reason": exit_reason,
                "entry_price": state.buy_price,
                "exit_price":  price,
                "btc": executed_qty,
                "received_usdt": received_usdt,
                "pnl_usdt": profit,
            })
            # reset trade state
            state.cash_balance += received_usdt
            state.btc_position = 0.0
            state.in_trade = False
            state.entry_timestamp = ""
            logger.info("Trade closed | pnl=$%.2f | new cash=$%.2f", profit, state.cash_balance)
        else:
            logger.info("Holding — drop from peak %.1f%%; trailing stop %.1f%%",
                       (state.peak_price - price) / state.peak_price * 100, state.current_trailing_stop * 100)
        save_state(state, state_path)
        return state

    # ── 4. We are flat — check entry conditions ──────────────────────────────
    if rsi > 70:  # overbought filter — same as backtest
        logger.info("Skipping entry — RSI %.1f > 70 (overbought).", rsi)
        save_state(state, state_path); return state

    if not (prev_ma10 <= prev_ma30 and ma10 > ma30):  # MA crossover signal
        logger.info("No crossover signal this candle.")
        save_state(state, state_path); return state

    if price < ma200_now or ma200_now < ma200_prev:  # trend filter — must be above MA200 AND MA200 rising
        logger.info("Trend filter rejected entry (price=$%.2f vs MA200=$%.2f).", price, ma200_now)
        save_state(state, state_path); return state

    # ── 5. ML gate ────────────────────────────────────────────────────────────
    features = compute_features_at_row(df, i)  # compute the same 29 features the model was trained on
    if features is None:
        logger.info("Not enough history to compute features — skipping.")
        save_state(state, state_path); return state
    ml_prob = ml_model.predict_proba(features)  # ML probability of price going up
    if ml_prob < ML_CONFIDENCE_THRESHOLD:
        logger.info("ML gate rejected entry — prob %.1f%% < threshold %.1f%%.", ml_prob * 100, ML_CONFIDENCE_THRESHOLD * 100)
        save_state(state, state_path); return state

    # ── 6. Regime detection — decide position size and trailing stop ─────────
    is_strong_uptrend = price >= ma200_now * (1 + BULL_REGIME_MA200_BUFFER) and ma200_now > ma200_prev
    position_pct      = BULL_POSITION_SIZE_PCT if is_strong_uptrend else POSITION_SIZE_PCT
    trailing_stop     = BULL_TRAILING_STOP_PCT if is_strong_uptrend else TRAILING_STOP_PCT

    # ── 7. Sync cash with what Binance actually says we have ─────────────────
    # Use the smaller of "what we think we have" vs "what's actually on the
    # exchange". Protects against state-file drift after manual interventions.
    real_cash = client.get_balance("USDT")
    safe_cash = min(state.cash_balance, real_cash)
    trade_cash = safe_cash * position_pct  # how many USDT to spend on this trade
    if trade_cash < 10.0:  # Binance has a minimum notional (~$10 per order)
        logger.warning("Trade size $%.2f below minimum — skipping.", trade_cash)
        save_state(state, state_path); return state

    # ── 8. Place BUY market order ────────────────────────────────────────────
    regime = "BULL" if is_strong_uptrend else "norm"
    logger.info("ENTRY → regime=%s size=%.0f%% trail=%.0f%% ML=%.0f%% — placing BUY $%.2f",
                regime, position_pct * 100, trailing_stop * 100, ml_prob * 100, trade_cash)
    order = client.place_market_order(symbol=symbol, side="BUY", quote_quantity=trade_cash)
    executed_qty = float(order.get("executedQty", 0))
    spent_usdt = float(order.get("cummulativeQuoteQty", trade_cash))

    # ── 9. Update state ──────────────────────────────────────────────────────
    state.cash_balance -= spent_usdt
    state.btc_position = executed_qty
    state.buy_price = price
    state.peak_price = price
    state.entry_timestamp = str(timestamp)
    state.current_trailing_stop = trailing_stop
    state.in_trade = True
    save_state(state, state_path)
    logger.info("Entered trade — bought %.6f BTC for $%.2f.", executed_qty, spent_usdt)
    return state
