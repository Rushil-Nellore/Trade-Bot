from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from quant_bot.config import (
    BULL_POSITION_SIZE_PCT,
    BULL_REGIME_MA200_BUFFER,
    BULL_TRAILING_STOP_PCT,
    FEE_RATE,
    HARD_STOP_PCT,
    ML_CONFIDENCE_THRESHOLD,
    POSITION_SIZE_PCT,
)
from quant_bot.ml import MLModel, compute_features_at_row
from quant_bot.models import SimulationResult
from quant_bot.strategy import should_exit_trade

logger = logging.getLogger(__name__)

_MIN_TRADE_VALUE = 10.0


def simulate(
    df: pd.DataFrame,
    initial_balance: float,
    trailing_stop_pct: float,
    position_size_pct: float = POSITION_SIZE_PCT,
    ml_model: MLModel | None = None,
    ml_confidence_threshold: float = ML_CONFIDENCE_THRESHOLD,
) -> SimulationResult:
    cash_balance = initial_balance
    btc_position = 0.0
    in_trade = False
    entry_timestamp: pd.Timestamp | None = None
    buy_price = 0.0
    peak_price = 0.0
    entry_value = 0.0
    current_trailing_stop = trailing_stop_pct
    bull_entries = 0
    normal_entries = 0

    trades: list[float] = []
    trade_log: list[dict[str, object]] = []
    balance_history: list[tuple[pd.Timestamp, float]] = []
    buy_signals: list[tuple[pd.Timestamp, float]] = []
    sell_signals: list[tuple[pd.Timestamp, float]] = []
    stop_signals: list[tuple[pd.Timestamp, float]] = []

    peak_balance = initial_balance
    max_drawdown = 0.0

    for i in range(1, len(df)):
        timestamp = df["timestamp"].iloc[i]
        price = df["close"].iloc[i]
        ma10 = df["MA_10"].iloc[i]
        ma30 = df["MA_30"].iloc[i]
        prev_ma10 = df["MA_10"].iloc[i - 1]
        prev_ma30 = df["MA_30"].iloc[i - 1]

        portfolio_value = cash_balance + (btc_position * price)

        if pd.isna(ma10) or pd.isna(ma30):
            balance_history.append((timestamp, portfolio_value))
            continue

        if not in_trade:
            rsi = df["RSI"].iloc[i]
            if pd.isna(rsi) or rsi > 70:
                balance_history.append((timestamp, portfolio_value))
                continue

            if prev_ma10 <= prev_ma30 and ma10 > ma30:
                ma200_now = df["MA_200"].iloc[i]
                ma200_prev = df["MA_200"].iloc[i - 10]
                if pd.isna(ma200_now) or pd.isna(ma200_prev):
                    balance_history.append((timestamp, portfolio_value))
                    continue
                if price < ma200_now or ma200_now < ma200_prev:
                    balance_history.append((timestamp, portfolio_value))
                    continue

                ml_prob: float | None = None
                if ml_model is not None and timestamp >= ml_model.train_cutoff_timestamp:
                    features = compute_features_at_row(df, i)
                    if features is None:
                        balance_history.append((timestamp, portfolio_value))
                        continue
                    ml_prob = ml_model.predict_proba(features)
                    if ml_prob < ml_confidence_threshold:
                        balance_history.append((timestamp, portfolio_value))
                        continue

                is_strong_uptrend = (
                    price >= ma200_now * (1 + BULL_REGIME_MA200_BUFFER)
                    and ma200_now > ma200_prev
                )
                if is_strong_uptrend:
                    active_position_pct = BULL_POSITION_SIZE_PCT
                    current_trailing_stop = BULL_TRAILING_STOP_PCT
                    bull_entries += 1
                else:
                    active_position_pct = position_size_pct
                    current_trailing_stop = trailing_stop_pct
                    normal_entries += 1

                trade_cash = cash_balance * active_position_pct
                if trade_cash < _MIN_TRADE_VALUE:
                    balance_history.append((timestamp, portfolio_value))
                    continue

                entry_value = trade_cash
                btc_position = (trade_cash * (1 - FEE_RATE)) / price
                cash_balance -= trade_cash
                buy_price = price
                entry_timestamp = timestamp
                peak_price = price
                in_trade = True
                buy_signals.append((timestamp, price))

                prob_str = f"{ml_prob:.0%}" if ml_prob is not None else "no-gate"
                regime = "BULL" if is_strong_uptrend else "norm"
                logger.info(
                    "BUY  $%s | %s | regime=%s | size=%.0f%% | trail=%.0f%% | ML=%s | cash=$%s",
                    f"{price:,.2f}",
                    timestamp,
                    regime,
                    active_position_pct * 100,
                    current_trailing_stop * 100,
                    prob_str,
                    f"{cash_balance:,.2f}",
                )

        else:
            peak_price = max(peak_price, price)
            exit_reason = should_exit_trade(
                price=price,
                buy_price=buy_price,
                peak_price=peak_price,
                trailing_stop_pct=current_trailing_stop,
                hard_stop_pct=HARD_STOP_PCT,
            )

            if exit_reason is None:
                portfolio_value = cash_balance + (btc_position * price)
                peak_balance = max(peak_balance, portfolio_value)
                max_drawdown = max(
                    max_drawdown,
                    (peak_balance - portfolio_value) / peak_balance,
                )
                balance_history.append((timestamp, portfolio_value))
                continue

            if exit_reason == "hard_stop":
                stop_signals.append((timestamp, price))
            else:
                sell_signals.append((timestamp, price))

            exit_value = btc_position * price * (1 - FEE_RATE)
            profit = exit_value - entry_value

            trade_log.append(
                {
                    "entry_time":        entry_timestamp,
                    "exit_time":         timestamp,
                    "exit_reason":       exit_reason,
                    "entry_price":       buy_price,
                    "exit_price":        price,
                    "position_size_btc": btc_position,
                    "gross_entry_value": entry_value,
                    "net_exit_value":    exit_value,
                    "pnl":               profit,
                    "return_pct":        profit / entry_value if entry_value else 0.0,
                }
            )

            cash_balance += exit_value
            btc_position = 0.0
            entry_timestamp = None
            trades.append(profit)
            in_trade = False

            peak_balance = max(peak_balance, cash_balance)
            max_drawdown = max(
                max_drawdown,
                (peak_balance - cash_balance) / peak_balance,
            )

            label = "STOP" if exit_reason == "hard_stop" else "SELL"
            logger.info(
                "%s $%s | pnl=$%s | balance=$%s",
                label,
                f"{price:,.2f}",
                f"{profit:,.2f}",
                f"{cash_balance:,.2f}",
            )

        portfolio_value = cash_balance + (btc_position * price)
        peak_balance = max(peak_balance, portfolio_value)
        max_drawdown = max(
            max_drawdown,
            (peak_balance - portfolio_value) / peak_balance,
        )
        balance_history.append((timestamp, portfolio_value))

    final_price = float(df["close"].iloc[-1])
    final_balance = cash_balance + (btc_position * final_price)

    sharpe_ratio = 0.0
    if len(balance_history) > 1:
        bal_values = np.array([v for _, v in balance_history], dtype=float)
        hourly_returns = np.diff(bal_values) / np.maximum(bal_values[:-1], 1e-10)
        mean_r = float(hourly_returns.mean())
        std_r  = float(hourly_returns.std())
        if std_r > 0:
            sharpe_ratio = (mean_r / std_r) * np.sqrt(24 * 365)

    first_price = float(df["close"].iloc[0])
    buy_and_hold_return = (final_price - first_price) / first_price if first_price else 0.0

    profitable = sum(1 for t in trades if t > 0)
    logger.info(
        "--- SUMMARY | trades=%d (bull=%d norm=%d) profitable=%d | drawdown=%.1f%% | Sharpe=%.2f | balance=$%s | buy-and-hold=%.1f%%",
        len(trades),
        bull_entries,
        normal_entries,
        profitable,
        max_drawdown * 100,
        sharpe_ratio,
        f"{final_balance:,.2f}",
        buy_and_hold_return * 100,
    )

    return SimulationResult(
        balance_history=balance_history,
        buy_signals=buy_signals,
        sell_signals=sell_signals,
        stop_signals=stop_signals,
        trades=trades,
        trade_log=trade_log,
        final_balance=final_balance,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        buy_and_hold_return=buy_and_hold_return,
    )
