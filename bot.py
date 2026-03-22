from quant_bot.config import DEFAULT_DAYS, DEFAULT_INTERVAL, DEFAULT_ML_HORIZON, DEFAULT_SYMBOL
from quant_bot.pipeline import run_full_pipeline


def main() -> None:
    df, result, ml_result = run_full_pipeline(
        symbol=DEFAULT_SYMBOL,
        interval=DEFAULT_INTERVAL,
        days=DEFAULT_DAYS,
        ml_horizon=DEFAULT_ML_HORIZON,
    )
    print(
        f"Fetched {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}"
    )
    print(f"Final balance: ${result.final_balance:,.2f}")
    print(
        "ML experiment accuracy: "
        f"{ml_result.metrics['accuracy']:.1%} "
        f"(baseline {ml_result.metrics['baseline_accuracy']:.1%})"
    )


if __name__ == "__main__":
    main()
