from quant_bot.config import DEFAULT_DAYS, DEFAULT_INTERVAL, DEFAULT_ML_HORIZON, DEFAULT_SYMBOL
from quant_bot.data import get_price_data
from quant_bot.ml import run_ml_experiment
from quant_bot.reporting import save_ml_report


def main() -> None:
    df = get_price_data(symbol=DEFAULT_SYMBOL, interval=DEFAULT_INTERVAL, days=DEFAULT_DAYS)
    result = run_ml_experiment(df, horizon=DEFAULT_ML_HORIZON)
    save_ml_report(result, path="ml_report.csv")
    print(f"ML accuracy: {result.metrics['accuracy']:.1%}")
    print(f"Baseline accuracy: {result.metrics['baseline_accuracy']:.1%}")
    print("\nTop features:")
    print(result.feature_importance.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
