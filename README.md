# BTC Quant Trading Bot

A portfolio-ready Python project that combines quantitative backtesting, a first-pass machine learning experiment, and a lightweight Streamlit dashboard for Bitcoin market analysis.

![BTC dashboard preview](dashboard.png)

## Why This Project Is Strong For Internships
- It shows end-to-end engineering, not just notebook experimentation.
- It mixes data analysis, trading logic, evaluation, visualization, and project structure.
- It includes a simple ML workflow that naturally connects to AI/ML coursework.
- It is organized like a real codebase with modules, tests, outputs, and repeatable commands.

## Features
- Binance hourly market-data ingestion for `BTCUSDT`
- Technical indicators: `MA_10`, `MA_30`, `MA_200`, `RSI(14)`
- Fee-aware long-only backtest with hard stop and trailing stop logic
- Trade export to `trade_log.csv`
- Chart export to `dashboard.png`
- ML experiment that predicts whether price will be higher after a future horizon
- Streamlit app for interactive viewing and reruns

## Project Structure
- `bot.py`: runs the full pipeline and regenerates outputs
- `ml_experiment.py`: runs only the ML experiment and exports `ml_report.csv`
- `app.py`: Streamlit dashboard for viewing and rerunning analysis
- `quant_bot/data.py`: data loading and indicator creation
- `quant_bot/backtest.py`: trading simulation
- `quant_bot/ml.py`: feature engineering and ML experiment
- `quant_bot/reporting.py`: chart and CSV exports
- `quant_bot/pipeline.py`: shared orchestration for scripts and app
- `tests/test_strategy.py`: regression tests for the exit-rule logic

## Backtest Strategy
- Entry: `MA_10` crosses above `MA_30`
- Trend filter: price stays above `MA_200` and `MA_200` is rising
- RSI filter: only buy when `RSI < 70`
- Hard stop: 4% below entry
- Trailing stop: 5% below the highest post-entry price
- Fees: 0.1% at entry and exit

## ML Upgrade
The ML module creates a supervised learning dataset from indicator-derived features and trains a lightweight logistic classifier from scratch with `numpy`.

Features used:
- 1-candle return
- 6-candle return
- 24-candle return
- short moving-average gap
- trend gap versus `MA_200`
- volume change
- scaled RSI

Target:
- `1` if price is higher after the prediction horizon
- `0` otherwise

This is intentionally simple and educational. It is meant to demonstrate feature engineering, train/test separation, baseline comparison, and model interpretation rather than claim production-grade predictive power.

## Streamlit Upgrade
The Streamlit app gives the project a more internship-friendly presentation layer:
- view the latest chart and trade log
- inspect ML metrics and feature weights
- rerun the full pipeline from a simple UI

Run it locally:

```bash
streamlit run app.py
```

## Setup
```bash
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python bot.py
python ml_experiment.py
streamlit run app.py
```

If PowerShell blocks activation, you can run without activating:

```bash
.\.venv\Scripts\python.exe bot.py
```

## Outputs
- `dashboard.png`: saved chart of price, signals, equity curve, and RSI
- `trade_log.csv`: one row per closed trade
- `ml_report.csv`: saved ML metrics and feature weights

## What This Project Demonstrates
- Python programming with modular code
- time-series data handling using `pandas`
- research thinking for strategy design
- basic machine learning workflow design
- experimentation discipline with saved outputs
- ability to make a project presentable for recruiters

## Limitations
- Results depend on the latest downloaded market data
- The ML model is intentionally simple and should not be treated as a trading edge
- The project is still a prototype, not a production system
- More tests, walk-forward validation, and benchmark comparisons would strengthen it further

## Resume-Friendly Description
Built a modular Bitcoin backtesting and ML analysis platform in Python using Binance market data, technical indicators, fee-aware trade simulation, CSV reporting, and a Streamlit dashboard for interactive analysis.
