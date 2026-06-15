BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
FEE_RATE = 0.001
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1h"
DEFAULT_DAYS = 365
DEFAULT_ML_HORIZON = 10 #predicting how the price is after 12 hours 
DEFAULT_TRAIN_RATIO = 0.8
INITIAL_BALANCE = 10_000
TRAILING_STOP_PCT = 0.05#if price rises to a peak and drops 5% then sell
HARD_STOP_PCT = 0.04
REQUEST_TIMEOUT = 15 # when asking data from binance wait for 15 seconds. if didnt get any response show error
TRADE_LOG_PATH = "trade_log.csv"
DASHBOARD_PATH = "dashboard.png"
ML_REPORT_PATH = "ml_report.csv"

# ── Position sizing ────────────────────────────────────────────────────────────
# Fraction of available cash to risk on each individual trade.
# 0.20 = 20 % per trade; keeps the rest in cash and prevents ruin on a single bad trade.
POSITION_SIZE_PCT = 0.20

# ── Regime-aware position sizing (NEW) ────────────────────────────────────────
# In confirmed uptrends (price comfortably above MA200 AND MA200 is rising) the
# bot uses a bigger position to capture more of bull-market upside.
# In choppy/bear regimes it falls back to the conservative POSITION_SIZE_PCT.
BULL_POSITION_SIZE_PCT = 0.40       # 40% of cash per trade in strong uptrends (vs 20% in choppy markets)
BULL_REGIME_MA200_BUFFER = 0.03     # price must be at least 3% above MA200 to count as "strong" uptrend (not just barely above)

# ── Regime-aware trailing stop (NEW) ──────────────────────────────────────────
# In a strong uptrend the bot uses a wider trailing stop so it does not get
# kicked out by routine 5% pullbacks during a trending bull run.
BULL_TRAILING_STOP_PCT = 0.10       # 10% trailing stop in strong uptrends (vs 5% in choppy markets)

# ── ML confidence gate ────────────────────────────────────────────────────────
# The XGBoost model must predict at least this probability of price going up
# before the backtester is allowed to enter a trade.
# Applied only AFTER the model's training cutoff (no lookahead).
ML_CONFIDENCE_THRESHOLD = 0.55

# ── Live trading configuration (NEW) ──────────────────────────────────────────
# These URLs let the live trader hit either Binance's free testnet (fake money,
# real-time prices) or the real exchange. ALWAYS test on testnet first.
# Sign up for free testnet API keys at: https://testnet.binance.vision
BINANCE_TESTNET_REST_URL = "https://testnet.binance.vision"  # Spot testnet — fake money, real market data
BINANCE_LIVE_REST_URL    = "https://api.binance.com"         # Production — REAL MONEY (do not use until paper-trading proves itself)
LIVE_STATE_PATH = "live_state.json"  # file where the live trader persists cash balance, position, peak price etc. between runs

# ── Historical market periods — presets for bot.py CLI and Streamlit dropdown ──
# Each entry maps a friendly name to a (days, end_date) combo.
# end_date=None means "ending today".
# Used by bot.py:    python bot.py bull_2021
# Used by app.py:    "Quick preset" dropdown in the sidebar
NOTABLE_PERIODS = {  # dict keyed by friendly name → dict with days and end_date
    "bull_2021":    {"days": 365,  "end_date": "2021-11-10"},  # peak of the 2021 bull run — BTC at ATH $69k
    "bear_2022":    {"days": 365,  "end_date": "2022-12-31"},  # 2022 crash year — Luna collapse + FTX collapse
    "bull_2023":    {"days": 365,  "end_date": "2024-03-14"},  # 2023 recovery — BTC $16k → new ATH $73k
    "bull_2024":    {"days": 365,  "end_date": "2025-01-20"},  # 2024-2025 bull run — BTC hits $109k
    "cycle_2years": {"days": 730,  "end_date": None},          # last 2 years from today — likely contains both up and down moves
    "cycle_3years": {"days": 1095, "end_date": None},          # last 3 years from today — full market cycle
}
