BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"  # URL of Binance's candlestick API — all price-data HTTP requests are sent to this address
FEE_RATE = 0.001  # trading fee as a decimal — 0.001 = 0.1%; Binance deducts this on every buy AND every sell
DEFAULT_SYMBOL = "BTCUSDT"  # default trading pair: BTC = Bitcoin, USDT = dollar-pegged stablecoin — prices are in USD
DEFAULT_INTERVAL = "1h"  # default candle size — each candle covers 1 hour of price action (open, high, low, close, volume)
DEFAULT_DAYS = 365  # number of calendar days of history to download from Binance by default
DEFAULT_ML_HORIZON = 10  # how many candles ahead the ML model predicts — 10 candles × 1 hour = 10 hours into the future
DEFAULT_TRAIN_RATIO = 0.8  # 80% of data trains the ML model; the remaining 20% is kept separate for honest evaluation
INITIAL_BALANCE = 10_000  # starting portfolio value in USD; underscores are Python's thousands separator — same as 10000
TRAILING_STOP_PCT = 0.05  # if price rises to a peak then drops 5% from that peak, exit the trade to lock in profits
HARD_STOP_PCT = 0.04  # if price falls 4% below the entry price, exit immediately — caps the maximum loss on any single trade
REQUEST_TIMEOUT = 15  # seconds to wait for a reply from Binance before timing out and retrying the request
TRADE_LOG_PATH = "trade_log.csv"  # output filename for the trade-by-trade backtest results (CSV opens in Excel/Sheets)
DASHBOARD_PATH = "dashboard.png"  # output filename for the chart image saved by matplotlib
ML_REPORT_PATH = "ml_report.csv"  # output filename for ML accuracy metrics and feature importance scores

# ── Position sizing ────────────────────────────────────────────────────────────
# Fraction of available cash to risk on each individual trade.
# 0.20 = 20% per trade; keeps the rest in cash and prevents a single bad trade from wiping the account.
POSITION_SIZE_PCT = 0.20  # each trade uses 20% of current cash; the other 80% stays safe in the portfolio

# ── ML confidence gate ────────────────────────────────────────────────────────
# The ML model must predict at least this probability of price going up
# before the backtester is allowed to enter a trade.
# Applied only AFTER the model's training cutoff — no lookahead bias.
ML_CONFIDENCE_THRESHOLD = 0.55  # require 55% confidence from the ML model; 50% = coin flip, so 55% demands a small real edge

# ── Historical market periods — use these as end_date + days in run_full_pipeline() ──
# Pass end_date="YYYY-MM-DD" and days=N to test any window in BTC history.
#
# BULL MARKETS (BTC was rising strongly):
#   2020-2021 mega bull run   → end_date="2021-11-10", days=365   (BTC: $10k  → $69k  +590%)
#   2023 recovery bull run    → end_date="2024-03-14", days=365   (BTC: $16k  → $73k  +356%)
#   2024-2025 bull run        → end_date="2025-01-20", days=365   (BTC: $40k  → $109k +172%)
#
# BEAR MARKETS (BTC was falling hard):
#   2022 crash (Luna/FTX)     → end_date="2022-12-31", days=365   (BTC: $46k  → $16k  -65%)
#   2025-2026 current period  → end_date="2026-05-30", days=365   (BTC: $107k → $75k  -30%)
#
# COMBINED (bull + bear in one window):
#   Full 2021-2022 cycle      → end_date="2022-12-31", days=730   (boom then bust)
#   Full 2023-2024 cycle      → end_date="2024-12-31", days=730   (recovery + ATH)
#   Recent 2 years            → end_date=None,         days=730   (last 2 years from today)
#   Recent 3 years            → end_date=None,         days=1095  (last 3 years from today)
#
# HOW TO USE in bot.py:
#   run_full_pipeline(days=365, end_date="2021-11-10")   ← 2021 bull market
#   run_full_pipeline(days=730)                           ← last 2 years (today)
NOTABLE_PERIODS = {  # dict of preset (days, end_date) tuples — pass **NOTABLE_PERIODS["bull_2021"] to run_full_pipeline()
    "bull_2021":    {"days": 365, "end_date": "2021-11-10"},  # peak of the 2021 bull run — BTC at all-time high $69k
    "bear_2022":    {"days": 365, "end_date": "2022-12-31"},  # full 2022 crash year — Luna collapse + FTX collapse
    "bull_2023":    {"days": 365, "end_date": "2024-03-14"},  # 2023 recovery — BTC goes from $16k to new ATH $73k
    "bull_2024":    {"days": 365, "end_date": "2025-01-20"},  # 2024-2025 bull run — BTC hits $109k
    "cycle_2years": {"days": 730, "end_date": None},          # last 2 years from today — likely contains both up and down moves
    "cycle_3years": {"days": 1095, "end_date": None},         # last 3 years from today — full market cycle
}
