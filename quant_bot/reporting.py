from __future__ import annotations

import pandas as pd

from quant_bot.config import DASHBOARD_PATH, INITIAL_BALANCE, TRADE_LOG_PATH
from quant_bot.ml import MLExperimentResult
from quant_bot.models import SimulationResult


def save_trade_log(result: SimulationResult, path: str = TRADE_LOG_PATH) -> None:
    trade_log_df = pd.DataFrame(result.trade_log)
    trade_log_df.to_csv(path, index=False)
    print(f"Trade log saved as {path}")


def save_ml_report(result: MLExperimentResult, path: str) -> None:
    metric_rows = pd.DataFrame(
        [{"section": "metrics", "name": key, "value": value} for key, value in result.metrics.items()]
    )
    feature_rows = result.feature_importance.copy()
    feature_rows["section"] = "feature_importance"
    feature_rows = feature_rows.rename(columns={"feature": "name", "weight": "value"})
    report = pd.concat(
        [metric_rows, feature_rows[["section", "name", "value"]]],
        ignore_index=True,
    )
    report.to_csv(path, index=False)
    print(f"ML report saved as {path}")


def plot(df: pd.DataFrame, result: SimulationResult, path: str = DASHBOARD_PATH) -> None:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(16, 12),
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )
    fig.patch.set_facecolor("#0f0f0f")
    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor("#0f0f0f")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    ax1.plot(df["timestamp"], df["close"], color="#ffffff", linewidth=0.8, label="BTC Price")
    ax1.plot(df["timestamp"], df["MA_10"], color="#f4a261", linewidth=1.2, label="MA10")
    ax1.plot(df["timestamp"], df["MA_30"], color="#2a9d8f", linewidth=1.2, label="MA30")
    ax1.plot(df["timestamp"], df["MA_200"], color="#e76f51", linewidth=1.2, linestyle="--", label="MA200")

    if result.buy_signals:
        bx, by = zip(*result.buy_signals)
        ax1.scatter(bx, by, color="#00ff88", marker="^", s=120, zorder=5, label="BUY")
    if result.sell_signals:
        sx, sy = zip(*result.sell_signals)
        ax1.scatter(sx, sy, color="#ff4444", marker="v", s=120, zorder=5, label="SELL")
    if result.stop_signals:
        stx, sty = zip(*result.stop_signals)
        ax1.scatter(stx, sty, color="#ff9900", marker="v", s=120, zorder=5, label="STOP")

    ax1.set_title("BTC Price + Signals", color="white", fontsize=13, pad=10)
    ax1.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"${value:,.0f}"))

    bal_times, bal_values = zip(*result.balance_history)
    ax2.plot(bal_times, bal_values, color="#a8dadc", linewidth=1.2)
    ax2.axhline(y=INITIAL_BALANCE, color="#555", linestyle="--", linewidth=0.8)
    ax2.fill_between(
        bal_times,
        INITIAL_BALANCE,
        bal_values,
        where=[value >= INITIAL_BALANCE for value in bal_values],
        color="#00ff88",
        alpha=0.15,
    )
    ax2.fill_between(
        bal_times,
        INITIAL_BALANCE,
        bal_values,
        where=[value < INITIAL_BALANCE for value in bal_values],
        color="#ff4444",
        alpha=0.15,
    )
    ax2.set_title("Portfolio Balance", color="white", fontsize=11, pad=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"${value:,.0f}"))

    ax3.plot(df["timestamp"], df["RSI"], color="#c77dff", linewidth=1.0)
    ax3.axhline(y=70, color="#ff4444", linestyle="--", linewidth=0.8, alpha=0.7)
    ax3.axhline(y=30, color="#00ff88", linestyle="--", linewidth=0.8, alpha=0.7)
    ax3.fill_between(df["timestamp"], 70, df["RSI"], where=df["RSI"] >= 70, color="#ff4444", alpha=0.2)
    ax3.fill_between(df["timestamp"], 30, df["RSI"], where=df["RSI"] <= 30, color="#00ff88", alpha=0.2)
    ax3.set_ylim(0, 100)
    ax3.set_title("RSI (14)", color="white", fontsize=11, pad=8)

    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")
    print(f"\nChart saved as {path}")
    plt.show()
