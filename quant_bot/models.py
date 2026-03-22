from __future__ import annotations

from dataclasses import dataclass

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

