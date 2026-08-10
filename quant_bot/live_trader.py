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
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd

from quant_bot.binance_client import BinanceClient
from quant_bot.config import (
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
from quant_bot.data import add_indicators
from quant_bot.ml import MLModel, compute_features_at_row
from quant_bot.strategy import should_exit_trade

logger = logging.getLogger(__name__)


@dataclass
class LiveState:
    cash_balance: float = INITIAL_BALANCE
    in_trade: bool = False
    btc_position: float = 0.0
    buy_price: float = 0.0
    peak_price: float = 0.0
    entry_timestamp: str = ""
    current_trailing_stop: float = TRAILING_STOP_PCT
    trade_history: list[dict] = field(default_factory=list)


def load_state(path: str = LIVE_STATE_PATH) -> LiveState:
    p = Path(path)
    if not p.exists():
        logger.info("No state file at %s, starting fresh.", path)
        return LiveState()
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded state: cash=$%.2f in_trade=%s", data.get("cash_balance", 0), data.get("in_trade", False))
    return LiveState(**data)


def save_state(state: LiveState, path: str = LIVE_STATE_PATH) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(asdict(state), f, indent=2, default=str)


def trade_once(
    client: BinanceClient,
    ml_model: MLModel,
    *,
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    state_path: str = LIVE_STATE_PATH,
) -> LiveState:
    state = load_state(state_path)

    candles = client.get_klines(symbol=symbol, interval=interval, limit=300)

    df = pd.DataFrame(candles, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = add_indicators(df).reset_index(drop=True)

    i = len(df) - 2
    if i < 200:
        logger.warning("Not enough candles loaded (%d) — skipping iteration.", len(df))
        return state

    timestamp = df["timestamp"].iloc[i]
    price = float(df["close"].iloc[i])
    ma10  = float(df["MA_10"].iloc[i])
    ma30  = float(df["MA_30"].iloc[i])
    ma200_now  = float(df["MA_200"].iloc[i])
    ma200_prev = float(df["MA_200"].iloc[i - 10])
    prev_ma10  = float(df["MA_10"].iloc[i - 1])
    prev_ma30  = float(df["MA_30"].iloc[i - 1])
    rsi = float(df["RSI"].iloc[i])

    logger.info("Decision tick @ %s | price=$%.2f | in_trade=%s | cash=$%.2f", timestamp, price, state.in_trade, state.cash_balance)

    if state.in_trade:
        state.peak_price = max(state.peak_price, price)
        exit_reason = should_exit_trade(
            price=price,
            buy_price=state.buy_price,
            peak_price=state.peak_price,
            trailing_stop_pct=state.current_trailing_stop,
            hard_stop_pct=HARD_STOP_PCT,
        )
        if exit_reason is not None:
            logger.info("Exit triggered (%s) at $%.2f — placing SELL order.", exit_reason, price)
            order = client.place_market_order(symbol=symbol, side="SELL", quantity=state.btc_position)
            executed_qty = float(order.get("executedQty", state.btc_position))
            received_usdt = float(order.get("cummulativeQuoteQty", executed_qty * price))
            profit = received_usdt - (state.btc_position * state.buy_price)

            state.trade_history.append({
                "entry_time": state.entry_timestamp,
                "exit_time":  str(timestamp),
                "exit_reason": exit_reason,
                "entry_price": state.buy_price,
                "exit_price":  price,
                "btc": executed_qty,
                "received_usdt": received_usdt,
                "pnl_usdt": profit,
            })
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

    if rsi > 70:
        logger.info("Skipping entry — RSI %.1f > 70 (overbought).", rsi)
        save_state(state, state_path); return state

    if not (prev_ma10 <= prev_ma30 and ma10 > ma30):
        logger.info("No crossover signal this candle.")
        save_state(state, state_path); return state

    if price < ma200_now or ma200_now < ma200_prev:
        logger.info("Trend filter rejected entry (price=$%.2f vs MA200=$%.2f).", price, ma200_now)
        save_state(state, state_path); return state

    features = compute_features_at_row(df, i)
    if features is None:
        logger.info("Not enough history to compute features — skipping.")
        save_state(state, state_path); return state
    ml_prob = ml_model.predict_proba(features)
    if ml_prob < ML_CONFIDENCE_THRESHOLD:
        logger.info("ML gate rejected entry — prob %.1f%% < threshold %.1f%%.", ml_prob * 100, ML_CONFIDENCE_THRESHOLD * 100)
        save_state(state, state_path); return state

    is_strong_uptrend = price >= ma200_now * (1 + BULL_REGIME_MA200_BUFFER) and ma200_now > ma200_prev
    position_pct      = BULL_POSITION_SIZE_PCT if is_strong_uptrend else POSITION_SIZE_PCT
    trailing_stop     = BULL_TRAILING_STOP_PCT if is_strong_uptrend else TRAILING_STOP_PCT

    real_cash = client.get_balance("USDT")
    safe_cash = min(state.cash_balance, real_cash)
    trade_cash = safe_cash * position_pct
    if trade_cash < 10.0:
        logger.warning("Trade size $%.2f below minimum — skipping.", trade_cash)
        save_state(state, state_path); return state

    regime = "BULL" if is_strong_uptrend else "norm"
    logger.info("ENTRY → regime=%s size=%.0f%% trail=%.0f%% ML=%.0f%% — placing BUY $%.2f",
                regime, position_pct * 100, trailing_stop * 100, ml_prob * 100, trade_cash)
    order = client.place_market_order(symbol=symbol, side="BUY", quote_quantity=trade_cash)
    executed_qty = float(order.get("executedQty", 0))
    spent_usdt = float(order.get("cummulativeQuoteQty", trade_cash))

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
