from __future__ import annotations  # allows newer type-hint syntax (e.g. list[str]) to work on Python 3.9 and earlier

from dataclasses import dataclass, field  # dataclass: decorator that auto-generates __init__, __repr__ etc.; field: lets you set default values on dataclass attributes

import pandas as pd  # pandas: the main Python library for working with tables of data — pd.Timestamp is a datetime type from it


@dataclass  # decorator that automatically writes boilerplate methods (init, repr, eq) so we only need to list the fields
class SimulationResult:  # a container that holds every output produced by one full backtest run
    balance_history: list[tuple[pd.Timestamp, float]]  # list of (timestamp, portfolio_value) pairs — one entry per candle — used to draw the balance chart
    buy_signals: list[tuple[pd.Timestamp, float]]  # list of (timestamp, price) for every BUY executed — used to plot green triangles on the chart
    sell_signals: list[tuple[pd.Timestamp, float]]  # list of (timestamp, price) for every normal SELL (trailing stop) — red triangles on the chart
    stop_signals: list[tuple[pd.Timestamp, float]]  # list of (timestamp, price) for every hard-stop exit — orange triangles on the chart
    trades: list[float]  # simple list of P&L (profit or loss) for each completed trade, e.g. [-18.64, 158.81, ...] — used to count wins/losses
    trade_log: list[dict[str, object]]  # detailed record of every trade as a dict with keys like entry_price, exit_price, pnl — saved to trade_log.csv
    final_balance: float  # total portfolio value (cash + any open BTC position) at the very last candle
    max_drawdown: float  # the worst peak-to-trough decline as a fraction, e.g. 0.03 = the portfolio fell 3% from its highest point at some point during the run
    sharpe_ratio: float = field(default=0.0)  # annualised risk-adjusted return — (mean hourly return / std of hourly returns) × √(24×365); field(default=0.0) means it defaults to 0 if not provided
    buy_and_hold_return: float = field(default=0.0)  # what a simple buy-on-day-1 hold-till-end strategy would have returned — used as a benchmark comparison
