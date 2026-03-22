from quant_bot.backtest import simulate
from quant_bot.config import DEFAULT_DAYS, DEFAULT_INTERVAL, DEFAULT_SYMBOL, INITIAL_BALANCE, TRAILING_STOP_PCT
from quant_bot.data import get_price_data
from quant_bot.reporting import plot, save_trade_log


def main() -> None:
    df = get_price_data(symbol=DEFAULT_SYMBOL, interval=DEFAULT_INTERVAL, days=DEFAULT_DAYS)
    print(
        f"Fetched {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}"
    )
    result = simulate(df, initial_balance=INITIAL_BALANCE, trailing_stop_pct=TRAILING_STOP_PCT)
    save_trade_log(result)
    plot(df, result)


if __name__ == "__main__":
    main()

