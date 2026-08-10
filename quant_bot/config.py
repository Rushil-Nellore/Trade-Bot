BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
FEE_RATE = 0.001
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVAL = "1h"
DEFAULT_DAYS = 365
DEFAULT_ML_HORIZON = 10
DEFAULT_TRAIN_RATIO = 0.8
INITIAL_BALANCE = 10_000
TRAILING_STOP_PCT = 0.05
HARD_STOP_PCT = 0.04
REQUEST_TIMEOUT = 15
TRADE_LOG_PATH = "trade_log.csv"
DASHBOARD_PATH = "dashboard.png"
ML_REPORT_PATH = "ml_report.csv"

POSITION_SIZE_PCT = 0.20

BULL_POSITION_SIZE_PCT = 0.40
BULL_REGIME_MA200_BUFFER = 0.03

BULL_TRAILING_STOP_PCT = 0.10

ML_CONFIDENCE_THRESHOLD = 0.55

BINANCE_TESTNET_REST_URL = "https://testnet.binance.vision"
BINANCE_LIVE_REST_URL    = "https://api.binance.com"
LIVE_STATE_PATH = "live_state.json"

NOTABLE_PERIODS = {
    "bull_2021":    {"days": 365,  "end_date": "2021-11-10"},
    "bear_2022":    {"days": 365,  "end_date": "2022-12-31"},
    "bull_2023":    {"days": 365,  "end_date": "2024-03-14"},
    "bull_2024":    {"days": 365,  "end_date": "2025-01-20"},
    "cycle_2years": {"days": 730,  "end_date": None},
    "cycle_3years": {"days": 1095, "end_date": None},
}
