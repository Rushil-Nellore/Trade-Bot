from __future__ import annotations

import pandas as pd

from quant_bot.config import FEE_RATE, HARD_STOP_PCT
from quant_bot.models import SimulationResult
from quant_bot.strategy import should_exit_trade


def simulate(df: pd.DataFrame, initial_balance: float, trailing_stop_pct: float) -> SimulationResult:
    cash_balance = initial_balance
    btc_position = 0.0
    in_trade = False
    entry_timestamp = None
    buy_price = 0.0
    peak_price = 0.0
    entry_value = 0.0
    trades = []
    trade_log = []
    balance_history = []
    buy_signals = []
    sell_signals = []
    stop_signals = []
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

                entry_value = cash_balance
                btc_position = (cash_balance * (1 - FEE_RATE)) / price
                cash_balance = 0.0
                buy_price = price
                entry_timestamp = timestamp
                peak_price = price
                in_trade = True
                buy_signals.append((timestamp, price))
                print(f"BUY  at ${price:,.2f} | {timestamp}")

        else:
            peak_price = max(peak_price, price)
            exit_reason = should_exit_trade(
                price=price,
                buy_price=buy_price,
                peak_price=peak_price,
                trailing_stop_pct=trailing_stop_pct,
                hard_stop_pct=HARD_STOP_PCT,
            )

            if exit_reason == "hard_stop":
                stop_signals.append((timestamp, price))
                print_prefix = "STOP"
            elif exit_reason == "trailing_stop":
                sell_signals.append((timestamp, price))
                print_prefix = "SELL"
            else:
                portfolio_value = cash_balance + (btc_position * price)
                peak_balance = max(peak_balance, portfolio_value)
                max_drawdown = max(max_drawdown, (peak_balance - portfolio_value) / peak_balance)
                balance_history.append((timestamp, portfolio_value))
                continue

            exit_value = btc_position * price * (1 - FEE_RATE)
            profit = exit_value - entry_value
            trade_log.append(
                {
                    "entry_time": entry_timestamp,
                    "exit_time": timestamp,
                    "exit_reason": exit_reason,
                    "entry_price": buy_price,
                    "exit_price": price,
                    "position_size_btc": btc_position,
                    "gross_entry_value": entry_value,
                    "net_exit_value": exit_value,
                    "pnl": profit,
                    "return_pct": profit / entry_value if entry_value else 0.0,
                }
            )
            cash_balance = exit_value
            btc_position = 0.0
            entry_timestamp = None
            trades.append(profit)
            in_trade = False
            peak_balance = max(peak_balance, cash_balance)
            max_drawdown = max(max_drawdown, (peak_balance - cash_balance) / peak_balance)
            print(f"{print_prefix} at ${price:,.2f} | pnl: ${profit:,.2f} | balance: ${cash_balance:,.2f}")

        portfolio_value = cash_balance + (btc_position * price)
        peak_balance = max(peak_balance, portfolio_value)
        max_drawdown = max(max_drawdown, (peak_balance - portfolio_value) / peak_balance)
        balance_history.append((timestamp, portfolio_value))

    final_price = df["close"].iloc[-1]
    final_balance = cash_balance + (btc_position * final_price)

    print("\n--- SUMMARY ---")
    print(f"Total trades : {len(trades)}")
    print(f"Profitable   : {sum(1 for trade in trades if trade > 0)}")
    print(f"Max drawdown : {max_drawdown:.1%}")
    print(f"Final balance: ${final_balance:,.2f}")

    return SimulationResult(
        balance_history=balance_history,
        buy_signals=buy_signals,
        sell_signals=sell_signals,
        stop_signals=stop_signals,
        trades=trades,
        trade_log=trade_log,
        final_balance=final_balance,
        max_drawdown=max_drawdown,
    )
