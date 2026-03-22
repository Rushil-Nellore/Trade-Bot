# BTC Quant Trading Bot

A Python backtesting project for a Bitcoin trend-following strategy using Binance hourly candles and a simple technical-analysis ruleset.

## What It Does
- Downloads historical `BTCUSDT` candle data from Binance.
- Calculates `MA_10`, `MA_30`, `MA_200`, and `RSI(14)`.
- Simulates a long-only strategy with trading fees, a hard stop, and a trailing stop.
- Saves a performance chart to `dashboard.png`.
- Saves every closed trade to `trade_log.csv`.

## Strategy Rules
- Entry: `MA_10` crosses above `MA_30`.
- Trend filter: price must be above `MA_200`, and `MA_200` must be rising.
- RSI filter: only enter when `RSI < 70`.
- Exit 1: 4% hard stop below entry.
- Exit 2: 5% trailing stop from the post-entry peak.
- Fees: 0.1% at entry and 0.1% at exit.

## Important Note On Results
This project is a prototype backtester, not a production trading system.

- Results will change depending on when you run the script because it pulls recent market data.
- Past results are not enough to validate a strategy.
- There are currently no automated tests for the backtest engine.
- The code is useful for learning and experimentation, but it should not be treated as investment advice.

## Setup
```bash
pip install -r requirements.txt
python bot.py
```

## Project Structure
- `bot.py`: thin entry point for running the backtest.
- `quant_bot/data.py`: market-data loading and indicator calculation.
- `quant_bot/backtest.py`: trading rules and portfolio simulation.
- `quant_bot/reporting.py`: chart export and trade log export.
- `tests/test_strategy.py`: first regression test for core exit logic.

## Outputs
- `dashboard.png`: price, signals, equity curve, and RSI chart.
- `trade_log.csv`: one row per closed trade with entry, exit, reason, size, and PnL.

## Project Quality Today
What is already good:
- Clear strategy logic.
- Real market data input.
- Visual output that makes the backtest easy to inspect.
- Trade logging for basic auditability.

What is still missing:
- Automated tests for indicators and order execution.
- Reproducible benchmark results checked into the repo.
- Walk-forward and out-of-sample evaluation.

## Next Improvements
### 1. Test Coverage
- Add unit tests for RSI, moving averages, entry logic, and stop logic.
- Add a small fixture dataset with known expected trades.

### 2. Better Research Workflow
- Save raw candle data locally for reproducible runs.
- Add CLI parameters for symbol, timeframe, and date range.

### 3. Better Risk Management
- Risk a fixed fraction of capital per trade instead of always going all-in.
- Add slippage assumptions and position sizing rules.

### 4. Stronger Validation
- Run the strategy across multiple market regimes.
- Compare against buy-and-hold and simpler baseline strategies.
