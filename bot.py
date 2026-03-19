import requests
import pandas as pd

def get_price_data(symbol="BTCUSDT", interval="1h", days=365):
    import time
    all_data = []
    # each request gets 1000 candles = 1000 hours
    # 1 year = 8760 hours, so we need ~9 requests
    end_time = int(time.time() * 1000)  # now in milliseconds
    hours_needed = days * 24
    candles_fetched = 0

    while candles_fetched < hours_needed:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": 1000,
            "endTime": end_time
        }
        response = requests.get("https://api.binance.com/api/v3/klines", params=params)
        data = response.json()
        if not data:
            break
        all_data = data + all_data  # prepend older data
        end_time = data[0][0] - 1  # move window back
        candles_fetched += len(data)

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df["close"] = pd.to_numeric(df["close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates("timestamp").reset_index(drop=True)
    df["MA_10"] = df["close"].rolling(window=10).mean()
    df["MA_30"] = df["close"].rolling(window=30).mean()
    df["MA_200"] = df["close"].rolling(window=200).mean()

    # RSI calculation
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
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
            rsi = df["RSI"].iloc[i]
            if pd.isna(rsi) or rsi > 70:
                continue  # overbought, skip
            if prev_ma10 <= prev_ma30 and ma10 > ma30:
                ma200 = df["MA_200"].iloc[i]
                if pd.isna(ma200) or price < ma200:
                    continue  # skip — downtrend, don't buy
                in_trade = True
                buy_price = price
                peak_price = price
                print(f"BUY  at ${price:,.2f} | {df['timestamp'].iloc[i]}")

        # SELL condition — trailing stop
        elif in_trade:
            peak_price = max(peak_price, price)
            drop_from_peak = (peak_price - price) / peak_price

        
            if price < buy_price * 0.96:  # 4% stop loss
                profit = price - buy_price
                balance += profit
                trades.append(profit)
                in_trade = False
                print(f"STOP at ${price:,.2f} | loss: ${profit:,.2f} | balance: ${balance:,.2f}")
                continue

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

df = get_price_data(days=365)
print(f"Fetched {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
simulate(df, balance=10000, trailing_stop_pct=0.05)