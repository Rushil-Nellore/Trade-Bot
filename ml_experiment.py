from quant_bot.config import DEFAULT_DAYS, DEFAULT_INTERVAL, DEFAULT_ML_HORIZON, DEFAULT_SYMBOL  # import the default settings from config
from quant_bot.data import get_price_data  # get_price_data(): fetches OHLCV candles from Binance and adds MA/RSI indicators
from quant_bot.ml import run_ml_experiment  # run_ml_experiment(): trains the sklearn logistic-regression model and evaluates it
from quant_bot.reporting import save_ml_report  # save_ml_report(): writes ML metrics and feature importance to a CSV file


def main() -> None:  # the standalone entry point for running only the ML experiment without the full backtest pipeline
    df = get_price_data(symbol=DEFAULT_SYMBOL, interval=DEFAULT_INTERVAL, days=DEFAULT_DAYS)  # fetch 365 days of BTCUSDT 1h candles with indicators — exactly the same data the full pipeline uses
    result = run_ml_experiment(df, horizon=DEFAULT_ML_HORIZON)  # train the sklearn LogisticRegression model on 80% of the data and evaluate on the remaining 20%
    save_ml_report(result, path="ml_report.csv")  # write the accuracy metrics and feature importance to ml_report.csv
    print(f"ML accuracy:       {result.metrics['accuracy']:.1%}")  # f-string with :.1% format: converts 0.53 to "53.0%" — shows how often the model predicted correctly on the test set
    print(f"Baseline accuracy: {result.metrics['baseline_accuracy']:.1%}")  # the accuracy of always-predicting the majority class — the model must beat this to be useful
    print("\nTop features:")  # blank line then header for the feature importance table
    print(result.feature_importance.head(5).to_string(index=False))  # .head(5): take the top 5 rows (most important features, already sorted); .to_string(index=False): convert to a plain text table without row numbers


if __name__ == "__main__":  # only call main() when this file is run directly (e.g. python ml_experiment.py) — not when imported
    main()
