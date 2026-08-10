import logging
import os
import sys

from quant_bot.config import (
    DASHBOARD_PATH,
    DEFAULT_DAYS,
    DEFAULT_INTERVAL,
    DEFAULT_ML_HORIZON,
    DEFAULT_SYMBOL,
    INITIAL_BALANCE,
    NOTABLE_PERIODS,
)
from quant_bot.pipeline import run_full_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


def _parse_args() -> tuple[int, str | None]:
    args = sys.argv[1:]

    if not args:
        return DEFAULT_DAYS, None

    if len(args) == 1 and args[0] in NOTABLE_PERIODS:
        preset = NOTABLE_PERIODS[args[0]]
        return preset["days"], preset["end_date"]

    if len(args) == 1 and args[0].isdigit():
        return int(args[0]), None

    if len(args) == 2 and args[1].isdigit():
        return int(args[1]), args[0]

    print("Usage:")
    print("  python bot.py                        — last 365 days (default)")
    print("  python bot.py <preset>               — named period from config (see NOTABLE_PERIODS)")
    print("  python bot.py <days>                 — last N days ending today")
    print("  python bot.py <YYYY-MM-DD> <days>    — N days ending on that date")
    print(f"\nAvailable presets: {', '.join(NOTABLE_PERIODS.keys())}")
    sys.exit(1)


def main() -> None:
    days, end_date = _parse_args()

    df, result, ml_result = run_full_pipeline(
        symbol=DEFAULT_SYMBOL,
        interval=DEFAULT_INTERVAL,
        days=days,
        end_date=end_date,
        ml_horizon=DEFAULT_ML_HORIZON,
    )

    sep = "-" * 48
    start = df["timestamp"].iloc[0].date()
    end   = df["timestamp"].iloc[-1].date()
    profitable = sum(1 for t in result.trades if t > 0)

    strat_pct = (result.final_balance / INITIAL_BALANCE - 1) * 100
    bnh_pct   = result.buy_and_hold_return * 100
    edge      = strat_pct - bnh_pct

    print(f"\n{sep}")
    print(f"  Period   : {start}  to  {end}  ({days} days)")
    print(f"  Candles  : {len(df):,}")
    print(f"  Trades   : {len(result.trades)}  (wins: {profitable}  losses: {len(result.trades)-profitable})")
    print(f"  Balance  : ${result.final_balance:,.2f}  ({strat_pct:+.1f}%)")
    print(f"  Drawdown : {result.max_drawdown:.1%}")
    print(f"  Sharpe   : {result.sharpe_ratio:.2f}")
    print(f"  BnH ret  : {bnh_pct:+.1f}%  (buy-and-hold over same period)")
    print(f"  Edge     : {edge:+.1f}pp  (strategy vs buy-and-hold)")
    print(f"  ML acc   : {ml_result.metrics['accuracy']:.1%}  (baseline {ml_result.metrics['baseline_accuracy']:.1%})")
    print(f"{sep}\n")

    os.startfile(DASHBOARD_PATH)


if __name__ == "__main__":
    main()
