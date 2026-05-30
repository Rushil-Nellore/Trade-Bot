from __future__ import annotations  # enables newer type-hint syntax on Python 3.9 and earlier

import logging  # Python's built-in logging library — structured timestamped output instead of bare print()

import numpy as np  # NumPy: fast numerical array library — used for Sharpe ratio calculation and array math
import pandas as pd  # pandas: data-table library — pd.DataFrame, pd.isna(), pd.Timestamp

from quant_bot.config import (  # import the constants this file needs from the central config
    FEE_RATE,                   # 0.001 = 0.1% — deducted on every buy and every sell
    HARD_STOP_PCT,              # 0.04 = 4% — exit if price drops 4% below entry
    ML_CONFIDENCE_THRESHOLD,    # 0.55 — ML model must be ≥55% confident before we buy
    POSITION_SIZE_PCT,          # 0.20 — only risk 20% of cash per trade
)
from quant_bot.ml import MLModel, compute_features_at_row  # MLModel: the trained classifier; compute_features_at_row: computes the 7 ML features for a single candle
from quant_bot.models import SimulationResult  # the dataclass container that holds all backtest outputs
from quant_bot.strategy import should_exit_trade  # the function that checks hard-stop and trailing-stop exit rules

logger = logging.getLogger(__name__)  # create a logger named "quant_bot.backtest" so every log line shows which file it came from

_MIN_TRADE_VALUE = 10.0  # minimum USD amount for a trade — avoids opening meaningless micro-positions when the account is nearly empty


def simulate(  # the main backtest engine — loops through every candle and simulates trading decisions
    df: pd.DataFrame,  # the full OHLCV + indicator DataFrame from get_price_data()
    initial_balance: float,  # starting cash in USD (e.g. 10000.0)
    trailing_stop_pct: float,  # sell if price falls this fraction below its peak (e.g. 0.05 = 5%)
    position_size_pct: float = POSITION_SIZE_PCT,  # fraction of cash to risk per trade — default 20%
    ml_model: MLModel | None = None,  # trained ML model for entry gating — None means no ML filter
    ml_confidence_threshold: float = ML_CONFIDENCE_THRESHOLD,  # minimum ML probability to allow entry — default 55%
) -> SimulationResult:  # returns a SimulationResult dataclass holding all outputs
    cash_balance = initial_balance  # current cash in USD — starts at the initial amount and changes with every trade
    btc_position = 0.0  # how much BTC we currently hold — 0.0 when no trade is open
    in_trade = False  # flag: True while a position is open, False when we are in cash
    entry_timestamp: pd.Timestamp | None = None  # the candle timestamp when we bought — None when not in a trade
    buy_price = 0.0  # the price at which we entered the current trade
    peak_price = 0.0  # the highest price BTC reached since we bought — updated every candle while in trade
    entry_value = 0.0  # how many USD we committed to the current trade (trade_cash at entry)

    trades: list[float] = []  # collects the P&L (profit or loss) of each completed trade as a float
    trade_log: list[dict[str, object]] = []  # detailed record of every trade — each trade is a dict; saved to trade_log.csv
    balance_history: list[tuple[pd.Timestamp, float]] = []  # (timestamp, portfolio_value) for every candle — used to draw the balance chart
    buy_signals: list[tuple[pd.Timestamp, float]] = []  # (timestamp, price) for every buy executed — drawn as green triangles on the chart
    sell_signals: list[tuple[pd.Timestamp, float]] = []  # (timestamp, price) for every trailing-stop sell — red triangles
    stop_signals: list[tuple[pd.Timestamp, float]] = []  # (timestamp, price) for every hard-stop exit — orange triangles

    peak_balance = initial_balance  # the highest portfolio value ever seen — used to calculate drawdown
    max_drawdown = 0.0  # the worst decline from peak as a fraction — updated continuously throughout the simulation

    for i in range(1, len(df)):  # loop through every row starting at index 1; we start at 1 (not 0) because some checks look at row i-1 (the previous candle)
        timestamp = df["timestamp"].iloc[i]  # .iloc[i]: integer-location — get this candle's timestamp
        price = df["close"].iloc[i]  # closing price of the current candle — used as the execution price for trades
        ma10 = df["MA_10"].iloc[i]  # 10-period moving average at the current candle
        ma30 = df["MA_30"].iloc[i]  # 30-period moving average at the current candle
        prev_ma10 = df["MA_10"].iloc[i - 1]  # MA10 at the previous candle — needed to detect a crossover
        prev_ma30 = df["MA_30"].iloc[i - 1]  # MA30 at the previous candle

        portfolio_value = cash_balance + (btc_position * price)  # total value = cash on hand + BTC holdings valued at current price

        if pd.isna(ma10) or pd.isna(ma30):  # pd.isna(): returns True if the value is NaN — MA10/MA30 are NaN for the first 30 candles (not enough history to compute them yet)
            balance_history.append((timestamp, portfolio_value))  # .append(): add this (time, value) pair to the list — we still record the balance even when we can't trade
            continue  # continue: skip the rest of this loop iteration and jump to the next candle

        if not in_trade:  # ── entry logic: only runs when we don't currently hold a position ──
            rsi = df["RSI"].iloc[i]  # RSI at the current candle — used as an overbought filter
            if pd.isna(rsi) or rsi > 70:  # skip if RSI is missing or above 70 (overbought — price may be due for a pullback, bad time to buy)
                balance_history.append((timestamp, portfolio_value))  # record balance before skipping
                continue  # jump to next candle

            if prev_ma10 <= prev_ma30 and ma10 > ma30:  # MA crossover signal: previous candle had MA10 ≤ MA30 (bearish/flat), current candle has MA10 > MA30 (bullish cross) — this is the "golden cross" buy signal
                ma200_now = df["MA_200"].iloc[i]  # 200-period MA at the current candle — used as the long-term trend filter
                ma200_prev = df["MA_200"].iloc[i - 10]  # MA200 ten candles ago — compare to check if the long-term trend is rising; NOTE: i is always ≥ 30 here (MA30 warmup ensures this) so i-10 is always positive
                if pd.isna(ma200_now) or pd.isna(ma200_prev):  # MA200 is NaN for the first 199 candles — skip if not ready
                    balance_history.append((timestamp, portfolio_value))
                    continue
                if price < ma200_now or ma200_now < ma200_prev:  # reject the signal if: price is below MA200 (we're in a downtrend) OR MA200 itself is declining (big-picture trend is falling)
                    balance_history.append((timestamp, portfolio_value))
                    continue

                ml_prob: float | None = None  # start with no ML probability — will be filled in if ML gating applies
                if ml_model is not None and timestamp >= ml_model.train_cutoff_timestamp:  # apply ML gate only if a model was provided AND we're past the training cutoff (after the cutoff = out-of-sample, safe to use)
                    features = compute_features_at_row(df, i)  # compute_features_at_row(): build the 7-feature dict for this candle using only backward-looking data
                    if features is None:  # None means not enough history to compute features yet — skip
                        balance_history.append((timestamp, portfolio_value))
                        continue
                    ml_prob = ml_model.predict_proba(features)  # predict_proba(): ask the trained model for P(price goes up in horizon candles) — returns 0.0–1.0
                    if ml_prob < ml_confidence_threshold:  # if model is not confident enough (below 55%), don't enter the trade
                        balance_history.append((timestamp, portfolio_value))
                        continue

                trade_cash = cash_balance * position_size_pct  # calculate how much USD to invest: 20% of current cash — prevents all-in bets
                if trade_cash < _MIN_TRADE_VALUE:  # if 20% of cash is less than $10, the position is too small to be worth opening
                    balance_history.append((timestamp, portfolio_value))
                    continue

                entry_value = trade_cash  # record how much USD we are committing to this trade — used later to calculate P&L
                btc_position = (trade_cash * (1 - FEE_RATE)) / price  # calculate how much BTC we receive: (1 - 0.001) deducts the 0.1% buy fee, then divide by price to convert USD to BTC
                cash_balance -= trade_cash  # subtract the invested amount from cash — the rest stays in cash earning 0%
                buy_price = price  # remember the entry price for stop-loss calculations
                entry_timestamp = timestamp  # remember when we entered for the trade log
                peak_price = price  # initialise the trailing peak to the entry price
                in_trade = True  # set the flag — we are now in an open position
                buy_signals.append((timestamp, price))  # record (time, price) for the chart's green triangle markers

                prob_str = f"{ml_prob:.0%}" if ml_prob is not None else "no-gate"  # format ML probability as a percentage string if we have one, otherwise "no-gate" (before the training cutoff)
                logger.info(  # log the buy event — visible in the terminal
                    "BUY  $%s | %s | ML=%s | cash_remaining=$%s",
                    f"{price:,.2f}",       # entry price formatted with comma separator and 2 decimal places
                    timestamp,             # the candle timestamp
                    prob_str,              # ML confidence or "no-gate"
                    f"{cash_balance:,.2f}",  # how much cash is still available after this trade
                )

        else:  # ── exit logic: only runs when we are already holding a BTC position ──
            peak_price = max(peak_price, price)  # max(): update the peak if today's price is higher than any price seen since entry — the trailing stop follows this peak upward
            exit_reason = should_exit_trade(  # ask strategy.py if we should exit; returns "hard_stop", "trailing_stop", or None
                price=price,
                buy_price=buy_price,
                peak_price=peak_price,
                trailing_stop_pct=trailing_stop_pct,
                hard_stop_pct=HARD_STOP_PCT,
            )

            if exit_reason is None:  # no exit signal — keep holding the position
                portfolio_value = cash_balance + (btc_position * price)  # recalculate total value with the latest price while in trade
                peak_balance = max(peak_balance, portfolio_value)  # update the all-time peak portfolio value
                max_drawdown = max(  # update max drawdown: how far we've fallen from the peak as a fraction
                    max_drawdown,
                    (peak_balance - portfolio_value) / peak_balance,  # (peak - current) / peak = drawdown fraction
                )
                balance_history.append((timestamp, portfolio_value))  # record this candle's portfolio value
                continue  # skip the rest of the loop body — nothing else to do this candle

            if exit_reason == "hard_stop":  # if we're exiting due to a hard stop (4% loss from entry)
                stop_signals.append((timestamp, price))  # record as a stop signal — shown as orange triangle on chart
            else:  # trailing_stop
                sell_signals.append((timestamp, price))  # record as a sell signal — shown as red triangle on chart

            exit_value = btc_position * price * (1 - FEE_RATE)  # proceeds from selling: BTC amount × current price, minus the 0.1% sell fee
            profit = exit_value - entry_value  # P&L = what we received back minus what we put in — positive = profit, negative = loss

            trade_log.append(  # append a detailed dict for this trade to the log list — saved to trade_log.csv
                {
                    "entry_time":        entry_timestamp,  # when we bought
                    "exit_time":         timestamp,         # when we sold
                    "exit_reason":       exit_reason,       # "hard_stop" or "trailing_stop"
                    "entry_price":       buy_price,         # price at which we bought BTC
                    "exit_price":        price,             # price at which we sold BTC
                    "position_size_btc": btc_position,      # how many BTC we held
                    "gross_entry_value": entry_value,       # USD committed at entry (before fee)
                    "net_exit_value":    exit_value,        # USD received at exit (after fee)
                    "pnl":               profit,            # profit or loss in USD
                    "return_pct":        profit / entry_value if entry_value else 0.0,  # P&L as a fraction of entry value; guard against division-by-zero with the if check
                }
            )

            cash_balance += exit_value  # add the sale proceeds back to cash — partial sizing means we still had 80% in cash, now we also get back the 20% trade portion
            btc_position = 0.0  # we no longer hold any BTC
            entry_timestamp = None  # clear the entry timestamp
            trades.append(profit)  # record the P&L for this trade in the simple list
            in_trade = False  # clear the flag — we are back to holding only cash

            peak_balance = max(peak_balance, cash_balance)  # update the all-time peak after receiving the exit proceeds
            max_drawdown = max(  # check if this exit produced a new worst drawdown
                max_drawdown,
                (peak_balance - cash_balance) / peak_balance,
            )

            label = "STOP" if exit_reason == "hard_stop" else "SELL"  # choose the log label based on which stop triggered
            logger.info(  # log the exit event
                "%s $%s | pnl=$%s | balance=$%s",
                label,
                f"{price:,.2f}",    # exit price
                f"{profit:,.2f}",   # profit/loss on this trade
                f"{cash_balance:,.2f}",  # total cash after exit
            )

        portfolio_value = cash_balance + (btc_position * price)  # recalculate portfolio value at end of candle (after any trade action)
        peak_balance = max(peak_balance, portfolio_value)  # update all-time peak portfolio value
        max_drawdown = max(  # update max drawdown one final time for this candle
            max_drawdown,
            (peak_balance - portfolio_value) / peak_balance,
        )
        balance_history.append((timestamp, portfolio_value))  # record this candle's final portfolio value for the chart

    final_price = float(df["close"].iloc[-1])  # .iloc[-1]: the very last row — the most recent closing price; float(): ensure it's a plain Python float
    final_balance = cash_balance + (btc_position * final_price)  # total value at end: any remaining cash plus any open BTC position valued at the last price

    # ── Sharpe ratio ──────────────────────────────────────────────────────────
    sharpe_ratio = 0.0  # default to 0 in case the balance history is too short
    if len(balance_history) > 1:  # need at least 2 data points to compute returns
        bal_values = np.array([v for _, v in balance_history], dtype=float)  # np.array(): convert the list of portfolio values to a NumPy array for fast math; list comprehension extracts just the value from each (timestamp, value) pair
        hourly_returns = np.diff(bal_values) / np.maximum(bal_values[:-1], 1e-10)  # np.diff(): compute (value[i+1] - value[i]) for every consecutive pair = hourly change; divide by previous value = return rate; np.maximum(..., 1e-10): protect against dividing by zero if balance ever hits 0
        mean_r = float(hourly_returns.mean())  # .mean(): average hourly return across all candles
        std_r  = float(hourly_returns.std())   # .std(): standard deviation of hourly returns — measures volatility/risk
        if std_r > 0:  # avoid division by zero if all returns are identical (e.g. perfectly flat market)
            sharpe_ratio = (mean_r / std_r) * np.sqrt(24 * 365)  # Sharpe = mean/std × annualisation factor; np.sqrt(24*365): convert hourly Sharpe to annualised Sharpe (24 hours × 365 days = 8760 hourly periods per year)

    # ── Buy-and-hold benchmark ────────────────────────────────────────────────
    first_price = float(df["close"].iloc[0])  # price at the very first candle
    buy_and_hold_return = (final_price - first_price) / first_price if first_price else 0.0  # (end - start) / start = simple return if you had bought on day 1 and held until the end; guard against zero first_price

    profitable = sum(1 for t in trades if t > 0)  # generator expression counts how many completed trades had positive P&L; sum() adds them up
    logger.info(  # print the final summary to the terminal
        "--- SUMMARY | trades=%d profitable=%d | drawdown=%.1f%% | Sharpe=%.2f | balance=$%s | buy-and-hold=%.1f%%",
        len(trades),             # total number of completed trades
        profitable,              # how many made money
        max_drawdown * 100,      # convert fraction to percentage (e.g. 0.03 → 3.0%)
        sharpe_ratio,            # annualised Sharpe ratio
        f"{final_balance:,.2f}", # final portfolio value formatted with commas
        buy_and_hold_return * 100,  # buy-and-hold return as percentage
    )

    return SimulationResult(  # pack all outputs into the dataclass and return to the caller
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
