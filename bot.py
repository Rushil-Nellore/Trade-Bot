import requests
import pandas as pd

def get_price_data(symbol="BTCUSDT", interval="1h", limit=100):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["close"] = pd.to_numeric(df["close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["MA_10"] = df["close"].rolling(window=10).mean()
    df["MA_30"] = df["close"].rolling(window=30).mean()
    return df

def simulate(df, balance=10000, trailing_stop_pct=0.03):
    in_trade = False
    buy_price = 0
    peak_price = 0
    trades = []

    for i in range(1, len(df)):
        price = df["close"].iloc[i]
        ma10  = df["MA_10"].iloc[i]
        ma30  = df["MA_30"].iloc[i]
        prev_ma10 = df["MA_10"].iloc[i-1]
        prev_ma30 = df["MA_30"].iloc[i-1]

        if pd.isna(ma10) or pd.isna(ma30):
            continue

        # BUY condition — crossover
        if not in_trade:
            if prev_ma10 <= prev_ma30 and ma10 > ma30:
                in_trade = True
                buy_price = price
                peak_price = price
                print(f"BUY  at ${price:,.2f} | {df['timestamp'].iloc[i]}")

        # SELL condition — trailing stop
        elif in_trade:
            peak_price = max(peak_price, price)
            drop_from_peak = (peak_price - price) / peak_price

            if drop_from_peak >= trailing_stop_pct:
                profit = price - buy_price
                balance += profit
                trades.append(profit)
                in_trade = False
                print(f"SELL at ${price:,.2f} | profit: ${profit:,.2f} | balance: ${balance:,.2f}")

    print(f"\n--- SUMMARY ---")
    print(f"Total trades : {len(trades)}")
    print(f"Profitable   : {sum(1 for t in trades if t > 0)}")
    print(f"Final balance: ${balance:,.2f}")
    return balance

df = get_price_data(limit=500)
simulate(df,balance=10000,trailing_stop_pct=0.05)