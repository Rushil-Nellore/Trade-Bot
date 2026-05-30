from __future__ import annotations  # enables newer type-hint syntax on Python 3.9 and earlier

import logging  # Python's built-in logging library — for timestamped output instead of print()

import pandas as pd  # pandas: data-table library — used to build DataFrames and write CSVs

from quant_bot.config import DASHBOARD_PATH, INITIAL_BALANCE, TRADE_LOG_PATH  # import the three constants this file uses
from quant_bot.ml import MLExperimentResult  # the ML result container — holds metrics and feature importance
from quant_bot.models import SimulationResult  # the backtest result container — holds trades, balance history, etc.

logger = logging.getLogger(__name__)  # create a logger named "quant_bot.reporting" so log lines identify this file


def save_trade_log(result: SimulationResult, path: str = TRADE_LOG_PATH) -> None:  # write the list of completed trades to a CSV file; default path from config
    if not result.trade_log:  # if the list is empty (no trades completed during the backtest)
        logger.warning("No completed trades — writing an empty trade log to %s.", path)  # logger.warning(): log at WARNING level — more visible than INFO; alerts the user that something unusual happened
    trade_log_df = pd.DataFrame(result.trade_log)  # pd.DataFrame(): convert the list of trade-dicts into a 2D table — each dict key becomes a column, each dict becomes a row
    trade_log_df.to_csv(path, index=False)  # .to_csv(): write the DataFrame to a CSV file; index=False: don't write the row numbers (0, 1, 2...) as an extra column — keeps the file clean
    logger.info("Trade log saved → %s", path)  # confirm save with an INFO-level log


def save_ml_report(result: MLExperimentResult, path: str) -> None:  # write ML accuracy metrics and feature importance to a CSV file
    metric_rows = pd.DataFrame(  # pd.DataFrame(): build a table from a list of dicts
        [{"section": "metrics", "name": key, "value": value} for key, value in result.metrics.items()]  # list comprehension: for each key/value pair in the metrics dict, create a row with section="metrics", the key as name, and the value
    )
    feature_rows = result.feature_importance.copy()  # copy(): take an independent copy of the feature importance DataFrame so we can modify it without affecting the original
    feature_rows["section"] = "feature_importance"  # add a "section" column filled with this string so the CSV has a consistent structure with the metrics section above
    feature_rows = feature_rows.rename(columns={"feature": "name", "weight": "value"})  # rename(): give the columns the same names as the metrics section so both sections can be stacked into one uniform CSV
    report = pd.concat(  # pd.concat(): stack multiple DataFrames vertically (one on top of the other) into a single DataFrame
        [metric_rows, feature_rows[["section", "name", "value"]]],  # select only the three shared columns from feature_rows before stacking
        ignore_index=True,  # ignore_index=True: renumber the combined rows 0, 1, 2, ... instead of keeping original row numbers
    )
    report.to_csv(path, index=False)  # .to_csv(): write the combined report to a CSV file; index=False: no row-number column
    logger.info("ML report saved → %s", path)  # confirm with an INFO log


def plot(df: pd.DataFrame, result: SimulationResult, path: str = DASHBOARD_PATH) -> None:  # render and save a 3-panel dark-theme chart; does NOT call plt.show() so it works in headless environments (Streamlit, CI servers)
    import matplotlib.dates as mdates  # mdates: matplotlib sub-module for date formatting on axis tick labels — imported lazily (inside the function) so matplotlib is only loaded when plotting is actually needed
    import matplotlib.pyplot as plt  # plt: the main matplotlib plotting interface — used to create figures, draw lines, scatter points, save images

    fig, (ax1, ax2, ax3) = plt.subplots(  # plt.subplots(): create a figure (fig) containing 3 vertically stacked subplots (ax1, ax2, ax3); returns the figure and a tuple of axes
        3,  # 3 rows of subplots
        1,  # 1 column of subplots
        figsize=(16, 12),  # total figure size in inches: 16 wide × 12 tall
        gridspec_kw={"height_ratios": [3, 1, 1]},  # height_ratios: the top panel is 3× taller than each of the two smaller panels
    )
    fig.patch.set_facecolor("#0f0f0f")  # fig.patch: the figure's background rectangle; set_facecolor(): set it to near-black (#0f0f0f) for the dark theme
    for ax in [ax1, ax2, ax3]:  # apply the same dark styling to all three panels
        ax.set_facecolor("#0f0f0f")  # set_facecolor(): paint the plot area background near-black
        ax.tick_params(colors="white")  # tick_params(colors="white"): make the axis tick labels (numbers on x and y axis) white so they're visible on black
        ax.xaxis.label.set_color("white")  # set_color("white"): make the x-axis label text white
        ax.yaxis.label.set_color("white")  # make the y-axis label text white
        for spine in ax.spines.values():  # ax.spines: the four border lines of each subplot; .values(): iterate over all four (top, bottom, left, right)
            spine.set_edgecolor("#333")  # set_edgecolor(): set the border line colour to dark grey (#333) so borders are subtly visible but not harsh

    ax1.plot(df["timestamp"], df["close"], color="#ffffff", linewidth=0.8, label="BTC Price")  # ax1.plot(): draw a line chart; x=timestamps, y=close prices; white line at 0.8 thickness; label= used in the legend
    ax1.plot(df["timestamp"], df["MA_10"], color="#f4a261", linewidth=1.2, label="MA10")  # draw the 10-period MA in orange
    ax1.plot(df["timestamp"], df["MA_30"], color="#2a9d8f", linewidth=1.2, label="MA30")  # draw the 30-period MA in teal
    ax1.plot(df["timestamp"], df["MA_200"], color="#e76f51", linewidth=1.2, linestyle="--", label="MA200")  # draw the 200-period MA in red-orange with a dashed line style to distinguish it

    if result.buy_signals:  # only plot buy markers if there are any — avoids error when zip(*[]) is called on an empty list
        bx, by = zip(*result.buy_signals)  # zip(*list_of_tuples): unzip a list of (time, price) pairs into two separate tuples: one of all times, one of all prices
        ax1.scatter(bx, by, color="#00ff88", marker="^", s=120, zorder=5, label="BUY")  # scatter(): plot individual points; marker="^" = upward triangle; s=120 = marker size; zorder=5 = draw on top of the lines
    if result.sell_signals:  # only plot if there are sell signals
        sx, sy = zip(*result.sell_signals)  # unzip sell signals into times and prices
        ax1.scatter(sx, sy, color="#ff4444", marker="v", s=120, zorder=5, label="SELL")  # marker="v" = downward triangle; red colour for sell
    if result.stop_signals:  # only plot if there are stop-loss exits
        stx, sty = zip(*result.stop_signals)  # unzip stop signals
        ax1.scatter(stx, sty, color="#ff9900", marker="v", s=120, zorder=5, label="STOP")  # orange downward triangle for hard stops

    ax1.set_title("BTC Price + Signals", color="white", fontsize=13, pad=10)  # set_title(): text above the panel; color="white": white text on dark background; pad=10: 10 pts of space between title and chart
    ax1.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)  # legend(): show a colour-coded key; facecolor: dark grey box; labelcolor: white text; fontsize: small to not cover the chart
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"${value:,.0f}"))  # FuncFormatter(): custom tick label function; lambda converts raw numbers to "$106,619" format on the y-axis

    if result.balance_history:  # only draw the balance chart if there is data
        bal_times, bal_values = zip(*result.balance_history)  # unzip the list of (timestamp, balance) pairs into two separate tuples
        ax2.plot(bal_times, bal_values, color="#a8dadc", linewidth=1.2)  # draw the portfolio balance curve in light blue
        ax2.axhline(y=INITIAL_BALANCE, color="#555", linestyle="--", linewidth=0.8)  # axhline(): draw a horizontal reference line at the $10,000 starting value so it's easy to see when the strategy is above or below its starting point
        ax2.fill_between(  # fill_between(): shade the area between two curves with a colour
            bal_times,
            INITIAL_BALANCE,  # lower boundary: the $10,000 starting line
            bal_values,       # upper boundary: the actual balance curve
            where=[v >= INITIAL_BALANCE for v in bal_values],  # where= condition: only fill this colour where balance is above $10,000
            color="#00ff88",  # green fill for profitable periods
            alpha=0.15,  # alpha=0.15: 85% transparent so the lines beneath are still visible
        )
        ax2.fill_between(  # second fill — below the starting balance
            bal_times,
            INITIAL_BALANCE,
            bal_values,
            where=[v < INITIAL_BALANCE for v in bal_values],  # only fill red where balance is below $10,000 (losing periods)
            color="#ff4444",  # red fill for loss periods
            alpha=0.15,
        )

    ax2.set_title(  # set the subtitle of the balance panel to also show key metrics at a glance
        f"Portfolio Balance  |  Sharpe {result.sharpe_ratio:.2f}"  # f-string: embed the Sharpe ratio formatted to 2 decimal places
        f"  |  Buy-and-hold {result.buy_and_hold_return:.1%}",  # embed buy-and-hold return formatted as a percentage
        color="white",
        fontsize=11,
        pad=8,
    )
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"${value:,.0f}"))  # format y-axis ticks as dollar amounts with commas

    ax3.plot(df["timestamp"], df["RSI"], color="#c77dff", linewidth=1.0)  # draw the RSI line in purple
    ax3.axhline(y=70, color="#ff4444", linestyle="--", linewidth=0.8, alpha=0.7)  # dashed red line at RSI=70 — overbought threshold
    ax3.axhline(y=30, color="#00ff88", linestyle="--", linewidth=0.8, alpha=0.7)  # dashed green line at RSI=30 — oversold threshold
    ax3.fill_between(df["timestamp"], 70, df["RSI"], where=df["RSI"] >= 70, color="#ff4444", alpha=0.2)  # shade red between RSI and the 70 line when RSI is overbought
    ax3.fill_between(df["timestamp"], 30, df["RSI"], where=df["RSI"] <= 30, color="#00ff88", alpha=0.2)  # shade green between RSI and the 30 line when RSI is oversold
    ax3.set_ylim(0, 100)  # set_ylim(): lock the y-axis range to 0–100 so RSI always fills the panel the same way
    ax3.set_title("RSI (14)", color="white", fontsize=11, pad=8)  # label the RSI panel

    for ax in [ax1, ax2, ax3]:  # apply the same x-axis date formatting to all three panels
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))  # DateFormatter("%b %Y"): format tick labels as "Jan 2025", "Feb 2025", etc.
        ax.xaxis.set_major_locator(mdates.MonthLocator())  # MonthLocator(): place one tick mark at the start of each calendar month
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")  # setp(): set properties on multiple objects at once; rotation=45: tilt tick labels 45° so they don't overlap; ha="right": right-align them at the tick position

    plt.tight_layout()  # tight_layout(): automatically adjust subplot spacing so titles, labels, and tick marks don't overlap each other
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f0f0f")  # savefig(): write the figure to disk as a PNG; dpi=150: medium-high resolution; bbox_inches="tight": crop out unnecessary whitespace around the figure; facecolor: keep the dark background in the saved file
    plt.close(fig)  # close(fig): release the figure from memory — important in loops and Streamlit to prevent memory leaks; deliberately NOT calling plt.show() so this works in headless server environments
    logger.info("Chart saved → %s", path)  # confirm the save with an INFO-level log message
