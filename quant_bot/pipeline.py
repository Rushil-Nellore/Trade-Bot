from __future__ import annotations  # enables newer type-hint syntax on Python 3.9 and earlier

import logging  # Python's built-in logging library — used to print timestamped progress messages

import pandas as pd  # pandas: data-table library — used here only for the return type annotation pd.DataFrame

from quant_bot.backtest import simulate  # simulate(): the main backtest engine that loops through candles and trades
from quant_bot.config import (  # import all the default constants used as function parameter defaults
    DASHBOARD_PATH,          # "dashboard.png" — default output path for the chart image
    DEFAULT_DAYS,            # 365 — default number of days of price history to fetch
    DEFAULT_INTERVAL,        # "1h" — default candle width
    DEFAULT_ML_HORIZON,      # 10 — default number of candles ahead for ML prediction
    DEFAULT_SYMBOL,          # "BTCUSDT" — default trading pair
    INITIAL_BALANCE,         # 10000 — default starting cash in USD
    ML_REPORT_PATH,          # "ml_report.csv" — default path for the ML metrics file
    POSITION_SIZE_PCT,       # 0.20 — default fraction of cash to risk per trade
    TRADE_LOG_PATH,          # "trade_log.csv" — default path for the trade log file
    TRAILING_STOP_PCT,       # 0.05 — default trailing stop percentage
)
from quant_bot.data import get_price_data  # get_price_data(): fetches OHLCV candles from Binance and adds indicators
from quant_bot.ml import MLExperimentResult, run_ml_experiment  # run_ml_experiment(): trains the ML model; MLExperimentResult: its return type
from quant_bot.models import SimulationResult  # SimulationResult: the dataclass that holds all backtest outputs
from quant_bot.reporting import plot, save_ml_report, save_trade_log  # the three output-saving functions

logger = logging.getLogger(__name__)  # create a logger named "quant_bot.pipeline" for this module


def run_full_pipeline(  # the single public function that wires together every step in the correct order
    symbol: str = DEFAULT_SYMBOL,  # which trading pair to analyse — defaults to "BTCUSDT"
    interval: str = DEFAULT_INTERVAL,  # candle width — defaults to "1h"
    days: int = DEFAULT_DAYS,  # how many days of history to fetch — defaults to 365
    end_date: str | None = None,  # optional historical end date as "YYYY-MM-DD" — None means use today; e.g. "2021-11-10" fetches the 365 days ending at the 2021 BTC ATH
    initial_balance: float = INITIAL_BALANCE,  # starting portfolio value — defaults to $10,000
    trailing_stop_pct: float = TRAILING_STOP_PCT,  # trailing-stop threshold — defaults to 5%
    position_size_pct: float = POSITION_SIZE_PCT,  # fraction of cash per trade — defaults to 20%
    ml_horizon: int = DEFAULT_ML_HORIZON,  # ML prediction horizon in candles — defaults to 10
    chart_path: str = DASHBOARD_PATH,  # where to save the chart PNG — defaults to "dashboard.png"
    trade_log_path: str = TRADE_LOG_PATH,  # where to save the trade log CSV — defaults to "trade_log.csv"
    ml_report_path: str = ML_REPORT_PATH,  # where to save the ML metrics CSV — defaults to "ml_report.csv"
) -> tuple[pd.DataFrame, SimulationResult, MLExperimentResult]:  # returns three objects: the price DataFrame, backtest results, ML results
    period_label = f"ending {end_date}" if end_date else "up to today"  # human-readable description of the time window for the log message
    logger.info("=== Pipeline start: %s %s %dd %s ===", symbol, interval, days, period_label)  # log a clearly visible start marker so the terminal output is easy to follow

    df = get_price_data(symbol=symbol, interval=interval, days=days, end_date=end_date)  # STEP 1: fetch historical OHLCV candles from Binance — end_date lets us fetch any historical window instead of just "right now"

    # STEP 2: train the ML model on the FIRST 80% of the data.
    # This must happen BEFORE the backtest so the trained model object exists
    # when simulate() needs it.  The model stores train_cutoff_timestamp so
    # the backtest can apply ML gating only on candles it was NOT trained on.
    ml_result = run_ml_experiment(df, horizon=ml_horizon)  # run_ml_experiment(): splits data, trains LogisticRegression via sklearn, evaluates on test split, returns MLExperimentResult including the trained MLModel

    # STEP 3: run the backtest with the trained model passed as an entry gate.
    # The simulate() function loops every candle but only asks the ML model
    # for a prediction on candles AFTER the training cutoff — zero lookahead.
    result = simulate(  # simulate(): loop through all candles, apply entry/exit rules, track balance — returns SimulationResult
        df,
        initial_balance=initial_balance,  # starting cash
        trailing_stop_pct=trailing_stop_pct,  # trailing stop %
        position_size_pct=position_size_pct,  # how much cash to risk per trade
        ml_model=ml_result.model,  # pass the trained MLModel so simulate() can call model.predict_proba() at each potential entry
    )

    save_trade_log(result, path=trade_log_path)  # STEP 4: write the list of trades to a CSV file
    save_ml_report(ml_result, path=ml_report_path)  # STEP 5: write ML accuracy metrics and feature importance to a CSV file
    plot(df, result, path=chart_path)  # STEP 6: render and save the 3-panel dark-theme chart (price/balance/RSI) to a PNG file

    logger.info("=== Pipeline complete ===")  # log a clearly visible end marker
    return df, result, ml_result  # return all three results to the caller (bot.py or app.py) so they can display summaries
