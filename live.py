"""Live (testnet) trading entry point — call this from cron/Task Scheduler hourly.

SAFETY: Defaults to TESTNET (fake money on real-time prices). To go LIVE with
real money, you must pass --live AND set BINANCE_LIVE=1 in your environment.
This double-confirmation prevents accidental real-money execution.

Setup (one-time):
    1. Get free testnet API keys from: https://testnet.binance.vision/
    2. Set environment variables:
           Windows PowerShell:   $env:BINANCE_API_KEY="..."   ; $env:BINANCE_API_SECRET="..."
           Bash:                 export BINANCE_API_KEY=...   ; export BINANCE_API_SECRET=...
    3. Train the model (just run `python bot.py` once — it saves nothing,
       so we retrain here every live call from scratch using the latest data).

Usage:
    python live.py                — one decision iteration (testnet)
    python live.py loop           — keep checking every hour (testnet)
    python live.py --live         — production (requires BINANCE_LIVE=1 env var)
"""
from __future__ import annotations  # newer type-hint syntax

import logging  # for structured timestamped log output
import os       # to read environment variables (API keys never go in code)
import sys      # to read command-line arguments
import time     # to sleep between iterations in loop mode

from quant_bot.binance_client import BinanceClient  # the REST client
from quant_bot.config import DEFAULT_DAYS, DEFAULT_INTERVAL, DEFAULT_ML_HORIZON, DEFAULT_SYMBOL  # defaults
from quant_bot.data import get_price_data  # fetches historical data for ML training
from quant_bot.live_trader import trade_once  # one-decision function we just wrote
from quant_bot.ml import run_ml_experiment  # trains the XGBoost model

logging.basicConfig(  # configure root logger so all modules' logs appear
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("live")


def _get_credentials() -> tuple[str, str]:  # read API keys from environment variables (never hardcode!)
    api_key = os.environ.get("BINANCE_API_KEY", "").strip()
    api_secret = os.environ.get("BINANCE_API_SECRET", "").strip()
    if not api_key or not api_secret:  # missing keys → print clear instructions
        print("ERROR: Missing API credentials.")
        print()
        print("Set environment variables first:")
        print("  PowerShell:  $env:BINANCE_API_KEY='...'  ;  $env:BINANCE_API_SECRET='...'")
        print("  Bash:        export BINANCE_API_KEY=...  ;  export BINANCE_API_SECRET=...")
        print()
        print("Get free testnet keys at: https://testnet.binance.vision/")
        sys.exit(1)  # exit with error code
    return api_key, api_secret


def main() -> None:
    args = sys.argv[1:]  # all command-line arguments after the script name
    use_live = "--live" in args  # default OFF — must pass --live to enable production
    loop_mode = "loop" in args  # if "loop" arg present → keep running, else single iteration

    if use_live:  # extra safety gate for production
        if os.environ.get("BINANCE_LIVE", "") != "1":
            print("ERROR: --live requires BINANCE_LIVE=1 in environment as a confirmation.")
            print("   This double-gate prevents accidental real-money execution.")
            sys.exit(1)
        logger.warning("!!! RUNNING WITH REAL MONEY — make sure you tested on testnet first !!!")

    api_key, api_secret = _get_credentials()  # load creds from env
    client = BinanceClient(api_key=api_key, api_secret=api_secret, testnet=not use_live)  # testnet=True unless --live

    # ── Train the ML model on the latest historical data ─────────────────────
    # We retrain on every fresh run (or every loop iteration) rather than persisting
    # the model — this keeps the model adapted to recent market conditions and
    # avoids the complexity of pickling/loading sklearn pipelines.
    logger.info("Training ML model on the most recent %d days of data...", DEFAULT_DAYS)
    df = get_price_data(symbol=DEFAULT_SYMBOL, interval=DEFAULT_INTERVAL, days=DEFAULT_DAYS)
    ml_result = run_ml_experiment(df, horizon=DEFAULT_ML_HORIZON)
    logger.info("Model trained — accuracy %.1f%% (baseline %.1f%%)",
                ml_result.metrics["accuracy"] * 100, ml_result.metrics["baseline_accuracy"] * 100)

    if loop_mode:  # keep running forever, checking every hour
        logger.info("Loop mode — will check every 3600 seconds (1 hour). Ctrl+C to stop.")
        while True:
            try:
                trade_once(client, ml_result.model)
            except Exception as exc:  # never let a single failed iteration kill the loop
                logger.exception("Iteration failed: %s", exc)
            time.sleep(3600)  # wait 1 hour before next check
    else:  # one iteration mode — call this from cron/Task Scheduler instead of using loop
        trade_once(client, ml_result.model)


if __name__ == "__main__":
    main()
