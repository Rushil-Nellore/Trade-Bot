from __future__ import annotations

import pandas as pd

from quant_bot.backtest import simulate
from quant_bot.config import (
    DASHBOARD_PATH,
    DEFAULT_DAYS,
    DEFAULT_INTERVAL,
    DEFAULT_ML_HORIZON,
    DEFAULT_SYMBOL,
    INITIAL_BALANCE,
    ML_REPORT_PATH,
    TRADE_LOG_PATH,
    TRAILING_STOP_PCT,
)
from quant_bot.data import get_price_data
from quant_bot.ml import MLExperimentResult, run_ml_experiment
from quant_bot.models import SimulationResult
from quant_bot.reporting import plot, save_ml_report, save_trade_log


def run_full_pipeline(
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    days: int = DEFAULT_DAYS,
    initial_balance: float = INITIAL_BALANCE,
    trailing_stop_pct: float = TRAILING_STOP_PCT,
    ml_horizon: int = DEFAULT_ML_HORIZON,
    chart_path: str = DASHBOARD_PATH,
    trade_log_path: str = TRADE_LOG_PATH,
    ml_report_path: str = ML_REPORT_PATH,
) -> tuple[pd.DataFrame, SimulationResult, MLExperimentResult]:
    df = get_price_data(symbol=symbol, interval=interval, days=days)
    result = simulate(df, initial_balance=initial_balance, trailing_stop_pct=trailing_stop_pct)
    ml_result = run_ml_experiment(df, horizon=ml_horizon)
    save_trade_log(result, path=trade_log_path)
    save_ml_report(ml_result, path=ml_report_path)
    plot(df, result, path=chart_path)
    return df, result, ml_result
