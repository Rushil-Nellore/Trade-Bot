from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class SimulationResult:
    balance_history: list[tuple[pd.Timestamp, float]]
    buy_signals: list[tuple[pd.Timestamp, float]]
    sell_signals: list[tuple[pd.Timestamp, float]]
    stop_signals: list[tuple[pd.Timestamp, float]]
    trades: list[float]
    trade_log: list[dict[str, object]]
    final_balance: float
    max_drawdown: float
    sharpe_ratio: float = field(default=0.0)
    buy_and_hold_return: float = field(default=0.0)
