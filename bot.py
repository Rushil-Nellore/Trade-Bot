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
def simulate(df, balance=10000, trailing_stop_pct=0.05):
    in_trade = False
    buy_price = 0
    peak_price = 0
    trades = []
    balance_history = []
    buy_signals = []
    sell_signals = []
    stop_signals = []
    peak_balance = balance
    max_drawdown = 0

    for i in range(1, len(df)):
        price = df["close"].iloc[i]
        ma10  = df["MA_10"].iloc[i]
        ma30  = df["MA_30"].iloc[i]
        prev_ma10 = df["MA_10"].iloc[i-1]
        prev_ma30 = df["MA_30"].iloc[i-1]

        if pd.isna(ma10) or pd.isna(ma30):
            balance_history.append((df["timestamp"].iloc[i], balance))
            continue

        if not in_trade:
            rsi = df["RSI"].iloc[i]
            if pd.isna(rsi) or rsi > 70:
                balance_history.append((df["timestamp"].iloc[i], balance))
                continue
            if prev_ma10 <= prev_ma30 and ma10 > ma30:
                ma200_now  = df["MA_200"].iloc[i]
                ma200_prev = df["MA_200"].iloc[i - 10]
                if pd.isna(ma200_now) or pd.isna(ma200_prev):
                    balance_history.append((df["timestamp"].iloc[i], balance))
                    continue
                if price < ma200_now or ma200_now < ma200_prev:
                    balance_history.append((df["timestamp"].iloc[i], balance))
                    continue
                in_trade = True
                buy_price = price
                peak_price = price
                buy_signals.append((df["timestamp"].iloc[i], price))
                print(f"BUY  at ${price:,.2f} | {df['timestamp'].iloc[i]}")

        elif in_trade:
            peak_price = max(peak_price, price)
            drop_from_peak = (peak_price - price) / peak_price

            if price < buy_price * 0.96:
                fee = price * 0.001
                profit = (price - buy_price) - fee
                balance += profit
                trades.append(profit)
                in_trade = False
                peak_balance = max(peak_balance, balance)
                max_drawdown = max(max_drawdown, (peak_balance - balance) / peak_balance)
                stop_signals.append((df["timestamp"].iloc[i], price))
                print(f"STOP at ${price:,.2f} | loss: ${profit:,.2f} | balance: ${balance:,.2f}")

            elif drop_from_peak >= trailing_stop_pct:
                fee = price * 0.001
                profit = (price - buy_price) - fee
                balance += profit
                trades.append(profit)
                in_trade = False
                peak_balance = max(peak_balance, balance)
                max_drawdown = max(max_drawdown, (peak_balance - balance) / peak_balance)
                sell_signals.append((df["timestamp"].iloc[i], price))
                print(f"SELL at ${price:,.2f} | profit: ${profit:,.2f} | balance: ${balance:,.2f}")

        balance_history.append((df["timestamp"].iloc[i], balance))

    print(f"\n--- SUMMARY ---")
    print(f"Total trades : {len(trades)}")
    print(f"Profitable   : {sum(1 for t in trades if t > 0)}")
    print(f"Max drawdown : {max_drawdown:.1%}")
    print(f"Final balance: ${balance:,.2f}")

    return balance_history, buy_signals, sell_signals, stop_signals

def plot(df, balance_history, buy_signals, sell_signals, stop_signals):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12),
                                         gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.patch.set_facecolor("#0f0f0f")
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#0f0f0f")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    # --- Price + MAs ---
    ax1.plot(df["timestamp"], df["close"],   color="#ffffff", linewidth=0.8, label="BTC Price")
    ax1.plot(df["timestamp"], df["MA_10"],   color="#f4a261", linewidth=1.2, label="MA10")
    ax1.plot(df["timestamp"], df["MA_30"],   color="#2a9d8f", linewidth=1.2, label="MA30")
    ax1.plot(df["timestamp"], df["MA_200"],  color="#e76f51", linewidth=1.2, label="MA200", linestyle="--")

    # buy/sell markers
    if buy_signals:
        bx, by = zip(*buy_signals)
        ax1.scatter(bx, by, color="#00ff88", marker="^", s=120, zorder=5, label="BUY")
    if sell_signals:
        sx, sy = zip(*sell_signals)
        ax1.scatter(sx, sy, color="#ff4444", marker="v", s=120, zorder=5, label="SELL")
    if stop_signals:
        stx, sty = zip(*stop_signals)
        ax1.scatter(stx, sty, color="#ff9900", marker="v", s=120, zorder=5, label="STOP")

    ax1.set_title("BTC Price + Signals", color="white", fontsize=13, pad=10)
    ax1.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # --- Portfolio Balance ---
    bal_times, bal_values = zip(*balance_history)
    ax2.plot(bal_times, bal_values, color="#a8dadc", linewidth=1.2)
    ax2.axhline(y=10000, color="#555", linestyle="--", linewidth=0.8)
    ax2.fill_between(bal_times, 10000, bal_values,
                     where=[v >= 10000 for v in bal_values],
                     color="#00ff88", alpha=0.15)
    ax2.fill_between(bal_times, 10000, bal_values,
                     where=[v < 10000 for v in bal_values],
                     color="#ff4444", alpha=0.15)
    ax2.set_title("Portfolio Balance", color="white", fontsize=11, pad=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # --- RSI ---
    ax3.plot(df["timestamp"], df["RSI"], color="#c77dff", linewidth=1.0)
    ax3.axhline(y=70, color="#ff4444", linestyle="--", linewidth=0.8, alpha=0.7)
    ax3.axhline(y=30, color="#00ff88", linestyle="--", linewidth=0.8, alpha=0.7)
    ax3.fill_between(df["timestamp"], 70, df["RSI"],
                     where=df["RSI"] >= 70, color="#ff4444", alpha=0.2)
    ax3.fill_between(df["timestamp"], 30, df["RSI"],
                     where=df["RSI"] <= 30, color="#00ff88", alpha=0.2)
    ax3.set_ylim(0, 100)
    ax3.set_title("RSI (14)", color="white", fontsize=11, pad=8)

    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig("dashboard.png", dpi=150, bbox_inches="tight",
                facecolor="#0f0f0f")
    print("\nChart saved as dashboard.png")
    plt.show()

df = get_price_data(days=365)
print(f"Fetched {len(df)} candles from {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
balance_history, buy_signals, sell_signals, stop_signals = simulate(df, balance=10000, trailing_stop_pct=0.05)
plot(df, balance_history, buy_signals, sell_signals, stop_signals)
df = get_price_data(days=365)

