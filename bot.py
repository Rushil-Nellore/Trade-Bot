import logging  # Python's built-in logging library — structured timestamped messages from all modules
import os  # Python's standard OS interface — used to open the saved chart in the Windows default image viewer
import sys  # Python's system library — used to read command-line arguments passed to the script

from quant_bot.config import (  # import constants from the central config
    DASHBOARD_PATH,
    DEFAULT_DAYS,
    DEFAULT_INTERVAL,
    DEFAULT_ML_HORIZON,
    DEFAULT_SYMBOL,
    INITIAL_BALANCE,
    NOTABLE_PERIODS,  # dict of preset (days, end_date) combos for different market conditions
)
from quant_bot.pipeline import run_full_pipeline

logging.basicConfig(  # configure the global logger — affects every module
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# ── HOW TO RUN DIFFERENT MARKET PERIODS ──────────────────────────────────────
#
#  Default (last 365 days, ending today — current bear market):
#    python bot.py
#
#  Named preset from NOTABLE_PERIODS in config.py:
#    python bot.py bull_2021       ← 2021 bull run  (BTC $10k → $69k)
#    python bot.py bear_2022       ← 2022 crash     (BTC $46k → $16k)
#    python bot.py bull_2023       ← 2023 recovery  (BTC $16k → $73k)
#    python bot.py bull_2024       ← 2024-25 bull   (BTC $40k → $109k)
#    python bot.py cycle_2years    ← last 2 years   (bull + bear combined)
#    python bot.py cycle_3years    ← last 3 years   (full market cycle)
#
#  Custom date + days (any window you want):
#    python bot.py 2021-11-10 365  ← 365 days ending 10 Nov 2021
#    python bot.py 2022-06-15 180  ← 180 days ending 15 Jun 2022
#    python bot.py 730             ← last 730 days ending today
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> tuple[int, str | None]:  # parse command-line arguments into (days, end_date) — returns defaults if nothing was passed
    args = sys.argv[1:]  # sys.argv: list of command-line tokens; [0] is the script name itself so we skip it with [1:]

    if not args:  # no arguments — use the defaults from config
        return DEFAULT_DAYS, None  # (365 days, no end_date = ending today)

    if len(args) == 1 and args[0] in NOTABLE_PERIODS:  # single argument that matches a preset name
        preset = NOTABLE_PERIODS[args[0]]  # look up the (days, end_date) dict for this preset name
        return preset["days"], preset["end_date"]  # unpack and return

    if len(args) == 1 and args[0].isdigit():  # single numeric argument — treat as custom days count
        return int(args[0]), None  # int(): convert the string "730" to the integer 730; no end_date = ending today

    if len(args) == 2 and args[1].isdigit():  # two arguments: end_date string + days number
        return int(args[1]), args[0]  # first arg is the date string, second is the days count

    # Unrecognised arguments — print help and exit
    print("Usage:")
    print("  python bot.py                        — last 365 days (default)")
    print("  python bot.py <preset>               — named period from config (see NOTABLE_PERIODS)")
    print("  python bot.py <days>                 — last N days ending today")
    print("  python bot.py <YYYY-MM-DD> <days>    — N days ending on that date")
    print(f"\nAvailable presets: {', '.join(NOTABLE_PERIODS.keys())}")
    sys.exit(1)  # sys.exit(1): exit the program with error code 1 (non-zero = failure)


def main() -> None:
    days, end_date = _parse_args()  # parse whatever the user typed — or use defaults

    df, result, ml_result = run_full_pipeline(
        symbol=DEFAULT_SYMBOL,
        interval=DEFAULT_INTERVAL,
        days=days,
        end_date=end_date,  # None = ending today; "YYYY-MM-DD" = ending on that historical date
        ml_horizon=DEFAULT_ML_HORIZON,
    )

    sep = "-" * 48
    start = df["timestamp"].iloc[0].date()   # first candle date
    end   = df["timestamp"].iloc[-1].date()  # last candle date
    profitable = sum(1 for t in result.trades if t > 0)  # count winning trades

    # ── Strategy return vs buy-and-hold ──────────────────────────────────────
    strat_pct = (result.final_balance / INITIAL_BALANCE - 1) * 100  # percentage gain/loss of the strategy
    bnh_pct   = result.buy_and_hold_return * 100  # buy-and-hold percentage
    edge      = strat_pct - bnh_pct  # how much the strategy beat (or lost to) buy-and-hold

    print(f"\n{sep}")
    print(f"  Period   : {start}  to  {end}  ({days} days)")
    print(f"  Candles  : {len(df):,}")
    print(f"  Trades   : {len(result.trades)}  (wins: {profitable}  losses: {len(result.trades)-profitable})")
    print(f"  Balance  : ${result.final_balance:,.2f}  ({strat_pct:+.1f}%)")
    print(f"  Drawdown : {result.max_drawdown:.1%}")
    print(f"  Sharpe   : {result.sharpe_ratio:.2f}")
    print(f"  BnH ret  : {bnh_pct:+.1f}%  (buy-and-hold over same period)")
    print(f"  Edge     : {edge:+.1f}pp  (strategy vs buy-and-hold)")  # pp = percentage points
    print(f"  ML acc   : {ml_result.metrics['accuracy']:.1%}  (baseline {ml_result.metrics['baseline_accuracy']:.1%})")
    print(f"{sep}\n")

    os.startfile(DASHBOARD_PATH)  # open the saved chart PNG in Windows default image viewer


if __name__ == "__main__":
    main()
