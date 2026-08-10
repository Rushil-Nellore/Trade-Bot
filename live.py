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
from __future__ import annotations

import logging
import os
import sys
import time

from quant_bot.binance_client import BinanceClient
from quant_bot.config import DEFAULT_DAYS, DEFAULT_INTERVAL, DEFAULT_ML_HORIZON, DEFAULT_SYMBOL
from quant_bot.data import get_price_data
from quant_bot.live_trader import trade_once
from quant_bot.ml import run_ml_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("live")


def _get_credentials() -> tuple[str, str]:
    api_key = os.environ.get("BINANCE_API_KEY", "").strip()
    api_secret = os.environ.get("BINANCE_API_SECRET", "").strip()
    if not api_key or not api_secret:
        print("ERROR: Missing API credentials.")
        print()
        print("Set environment variables first:")
        print("  PowerShell:  $env:BINANCE_API_KEY='...'  ;  $env:BINANCE_API_SECRET='...'")
        print("  Bash:        export BINANCE_API_KEY=...  ;  export BINANCE_API_SECRET=...")
        print()
        print("Get free testnet keys at: https://testnet.binance.vision/")
        sys.exit(1)
    return api_key, api_secret


def main() -> None:
    args = sys.argv[1:]
    use_live = "--live" in args
    loop_mode = "loop" in args

    if use_live:
        if os.environ.get("BINANCE_LIVE", "") != "1":
            print("ERROR: --live requires BINANCE_LIVE=1 in environment as a confirmation.")
            print("   This double-gate prevents accidental real-money execution.")
            sys.exit(1)
        logger.warning("!!! RUNNING WITH REAL MONEY — make sure you tested on testnet first !!!")

    api_key, api_secret = _get_credentials()
    client = BinanceClient(api_key=api_key, api_secret=api_secret, testnet=not use_live)

    logger.info("Training ML model on the most recent %d days of data...", DEFAULT_DAYS)
    df = get_price_data(symbol=DEFAULT_SYMBOL, interval=DEFAULT_INTERVAL, days=DEFAULT_DAYS)
    ml_result = run_ml_experiment(df, horizon=DEFAULT_ML_HORIZON)
    logger.info("Model trained — accuracy %.1f%% (baseline %.1f%%)",
                ml_result.metrics["accuracy"] * 100, ml_result.metrics["baseline_accuracy"] * 100)

    if loop_mode:
        logger.info("Loop mode — will check every 3600 seconds (1 hour). Ctrl+C to stop.")
        while True:
            try:
                trade_once(client, ml_result.model)
            except Exception as exc:
                logger.exception("Iteration failed: %s", exc)
            time.sleep(3600)
    else:
        trade_once(client, ml_result.model)


if __name__ == "__main__":
    main()
