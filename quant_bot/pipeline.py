from __future__ import annotations

import logging

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
    POSITION_SIZE_PCT,
    TRADE_LOG_PATH,
    TRAILING_STOP_PCT,
)
from quant_bot.data import get_price_data
from quant_bot.ml import MLExperimentResult, run_ml_experiment
from quant_bot.models import SimulationResult
from quant_bot.reporting import plot, save_ml_report, save_trade_log

logger = logging.getLogger(__name__)


def run_full_pipeline(
    symbol: str = DEFAULT_SYMBOL,
    interval: str = DEFAULT_INTERVAL,
    days: int = DEFAULT_DAYS,
    end_date: str | None = None,
    initial_balance: float = INITIAL_BALANCE,
    trailing_stop_pct: float = TRAILING_STOP_PCT,
    position_size_pct: float = POSITION_SIZE_PCT,
    ml_horizon: int = DEFAULT_ML_HORIZON,
    chart_path: str = DASHBOARD_PATH,
    trade_log_path: str = TRADE_LOG_PATH,
    ml_report_path: str = ML_REPORT_PATH,
) -> tuple[pd.DataFrame, SimulationResult, MLExperimentResult]:
    period_label = f"ending {end_date}" if end_date else "up to today"
    logger.info("=== Pipeline start: %s %s %dd %s ===", symbol, interval, days, period_label)

    df = get_price_data(symbol=symbol, interval=interval, days=days, end_date=end_date)

    ml_result = run_ml_experiment(df, horizon=ml_horizon)

    result = simulate(
        df,
        initial_balance=initial_balance,
        trailing_stop_pct=trailing_stop_pct,
        position_size_pct=position_size_pct,
        ml_model=ml_result.model,
    )

    save_trade_log(result, path=trade_log_path)
    save_ml_report(ml_result, path=ml_report_path)
    plot(df, result, path=chart_path)

    logger.info("=== Pipeline complete ===")
    return df, result, ml_result
