# BTC Quant Trading Bot

A Python-based algorithmic trading bot for Bitcoin using technical analysis.

## Strategy
- **Entry:** MA10/MA30 crossover with MA200 trend filter + RSI < 70
- **Exit:** 5% trailing stop or 4% hard stop loss
- **Fees:** 0.1% per trade (Binance standard)

## Results (Mar 2025 - Mar 2026)
- Starting balance: $10,000
- Final balance: $34,817
- Total trades: 13
- Win rate: 54%
- Max drawdown: 22.8%

## Setup
pip install requests pandas matplotlib
python bot.py

## Indicators Used
- MA10 / MA30 — momentum crossover signal
- MA200 — long term trend filter
- RSI (14) — overbought filter

## Future Improvements

### 1. ML-Based Signal Generation
Replace rule-based MA crossover signals with a trained ML model:
- Use LSTM (Long Short-Term Memory) neural network to predict price direction
- Features: MA10, MA30, MA200, RSI, volume, price momentum
- Label: 1 (price up next 10 hours), 0 (price down)
- This directly connects to Andrew Ng's ML Specialization coursework

### 2. Multi-Asset Trading
- Run the bot on ETH, SOL, BNB simultaneously
- Allocate capital across assets based on signal strength

### 3. Position Sizing
- Risk only 1-2% of portfolio per trade instead of going all-in
- Kelly Criterion for optimal bet sizing

### 4. Walk-Forward Testing
- Test on 2022-2023 bear market data
- Validate strategy isn't overfitted to 2025-2026

### 5. Live Paper Trading
- Connect to Binance Testnet (fake money, real market conditions)
- Run bot 24/7 on a Raspberry Pi or cloud server (AWS/GCP free tier)

### 6. Telegram Alerts
- Send a Telegram message every time the bot buys or sells
- Monitor it from your phone in real time