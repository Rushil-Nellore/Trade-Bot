from __future__ import annotations  # enables newer type-hint syntax on Python 3.9 and earlier

import logging  # Python's built-in logging library — forward logs to Streamlit's terminal
from pathlib import Path  # pathlib.Path: object-oriented file path handling — cleaner than string manipulation for checking if files exist

import pandas as pd  # pandas: data-table library — used to load and display CSVs
import streamlit as st  # streamlit: the web dashboard framework — every st.* call renders a UI element in the browser

from quant_bot.config import (  # import constants from config
    DEFAULT_DAYS,
    DEFAULT_INTERVAL,
    DEFAULT_ML_HORIZON,
    DEFAULT_SYMBOL,
    INITIAL_BALANCE,
    NOTABLE_PERIODS,  # dict of preset (days, end_date) combos for famous market periods — used to populate the Quick preset dropdown
)
from quant_bot.pipeline import run_full_pipeline  # the function that runs the entire analysis pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")  # configure logging so all module log messages appear in Streamlit's terminal with timestamps

st.set_page_config(page_title="BTC Quant Dashboard", page_icon="📈", layout="wide")  # set_page_config(): configure the browser tab — title shown in the tab, icon emoji, layout="wide" uses the full browser width


def load_csv(path: str) -> pd.DataFrame:  # helper function: safely load a CSV file — returns an empty DataFrame if the file doesn't exist yet
    file_path = Path(path)  # Path(path): convert the string path to a Path object which has .exists(), .read_text() etc. as methods
    if not file_path.exists():  # .exists(): returns True if the file is present on disk — False if the pipeline hasn't been run yet
        return pd.DataFrame()  # pd.DataFrame(): return an empty DataFrame (zero rows and columns) as a safe default
    return pd.read_csv(file_path)  # pd.read_csv(): read the CSV file from disk into a DataFrame — each column in the CSV becomes a DataFrame column


st.title("BTC Quant Trading Bot")  # st.title(): render a large H1 heading at the top of the page
st.caption(  # st.caption(): render small grey descriptive text below the title
    "MA cross-over strategy gated by a logistic-regression ML model. "
    "The model is trained on the first 80 % of data; ML gating is only active on the remaining 20 %."
)

with st.sidebar:  # st.sidebar: context manager that places all contained UI elements in the collapsible left sidebar
    st.header("Run Settings")  # st.header(): H2-level heading inside the sidebar
    symbol = st.text_input("Symbol", value=DEFAULT_SYMBOL)  # st.text_input(): a text box — user can type a symbol; value= sets the pre-filled default text
    interval = st.selectbox("Interval", options=["1h", "4h", "1d"], index=0)  # st.selectbox(): a dropdown menu; options= is the list of choices; index=0 pre-selects "1h"
    days = st.slider("Lookback days", min_value=90, max_value=1095, value=DEFAULT_DAYS, step=30)  # st.slider(): draggable slider; max raised to 1095 (3 years) so you can test full market cycles

    st.markdown("---")  # st.markdown("---"): renders a horizontal divider line in the sidebar
    st.subheader("Historical Period")  # st.subheader(): H3 heading
    st.caption("Leave end date blank to use today. Set it to test a past window.")  # small hint text

    preset_options = ["(none — use end date below)"] + list(NOTABLE_PERIODS.keys())  # build the dropdown list: first option means "manual", rest are the preset names from config
    selected_preset = st.selectbox(  # dropdown to pick a named market period
        "Quick preset",
        options=preset_options,  # the list built above
        index=0,  # start with "(none)"
        help="Presets automatically set days + end date for well-known market periods",  # help=: tooltip shown when hovering over the widget
    )

    if selected_preset != preset_options[0]:  # if the user picked an actual preset (not the "(none)" placeholder)
        preset_vals = NOTABLE_PERIODS[selected_preset]  # look up the (days, end_date) dict for this preset
        days = preset_vals["days"]  # override the days slider value with the preset's days
        preset_end = preset_vals["end_date"]  # the preset's end date (may be None for "today" presets)
        end_date_input = st.text_input(  # show the end date in a text box but pre-fill it so the user can see what they selected
            "End date (YYYY-MM-DD)",
            value=preset_end if preset_end else "",  # blank if the preset uses today
            disabled=True,  # disabled=True: read-only — the preset controls the value, user shouldn't type here
        )
        end_date = preset_end  # use the preset's end date (None means "up to today")
    else:  # no preset selected — let the user type a custom date
        end_date_input = st.text_input(  # free-text box for a custom end date
            "End date (YYYY-MM-DD)",
            value="",  # blank by default = fetch up to today
            placeholder="e.g. 2021-11-10",  # placeholder=: ghost text shown when the box is empty
            help="Leave blank for the most recent data. Enter a past date to analyse a historical window.",
        )
        end_date = end_date_input.strip() if end_date_input.strip() else None  # .strip(): remove accidental spaces; if the box is empty, use None (= today)

    st.markdown("---")  # another divider line
    ml_horizon = st.slider("ML horizon (candles ahead)", min_value=3, max_value=48, value=DEFAULT_ML_HORIZON)  # slider for the ML prediction horizon
    run_button = st.button("Run fresh analysis", type="primary")  # primary blue button — returns True when clicked

if run_button:  # this block executes only when the user has clicked the button — run_button is True for just that one script re-run
    with st.spinner("Fetching market data, training ML model, running backtest…"):  # st.spinner(): shows an animated loading spinner with the given message while the indented code block executes
        df, result, ml_result = run_full_pipeline(  # run the entire pipeline — fetches Binance data, trains ML, runs backtest, saves files
            symbol=symbol,
            interval=interval,
            days=days,
            end_date=end_date,  # None = fetch up to today; "YYYY-MM-DD" = fetch a historical window
            ml_horizon=ml_horizon,
        )
    st.success("Analysis complete.")  # st.success(): render a green success banner when the pipeline finishes
    st.session_state["latest_metrics"] = {  # st.session_state: a dict that persists across Streamlit script re-runs — stores the metrics so they still show even after the page refreshes
        "candles":             len(df),
        "final_balance":       result.final_balance,
        "max_drawdown":        result.max_drawdown,
        "sharpe_ratio":        result.sharpe_ratio,
        "buy_and_hold_return": result.buy_and_hold_return,
        "ml_accuracy":         ml_result.metrics["accuracy"],
        "baseline_accuracy":   ml_result.metrics["baseline_accuracy"],
    }

trade_log_df = load_csv("trade_log.csv")  # try to load the previously saved trade log — may be empty if pipeline hasn't run yet
ml_report_df = load_csv("ml_report.csv")  # try to load the previously saved ML report
chart_path = Path("dashboard.png")  # Path object for the saved chart image — used to check existence and display

metrics = st.session_state.get("latest_metrics", {})  # .get(key, default): retrieve the metrics dict from session state; return empty dict {} if the pipeline hasn't run this session yet

col1, col2, col3, col4, col5 = st.columns(5)  # st.columns(5): split the page into 5 equal-width columns and unpack each into a variable
col1.metric("Candles", f"{int(metrics['candles']):,}" if metrics else "—")  # .metric(): renders a big bold number with a label above it; conditional expression shows "—" placeholder if no run has happened
col2.metric(  # Final Balance card with a delta showing % gain/loss vs starting balance
    "Final Balance",
    f"${metrics['final_balance']:,.2f}" if metrics else "—",  # main value: formatted as dollar amount
    delta=f"{(metrics['final_balance'] / INITIAL_BALANCE - 1):.1%} vs start" if metrics else None,  # BUG FIX: now uses INITIAL_BALANCE from config instead of hardcoded 10_000; delta= shows a green/red sub-number
)
col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}" if metrics else "—")  # worst peak-to-trough loss as a percentage
col4.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}" if metrics else "—")  # annualised risk-adjusted return metric
col5.metric(  # ML Accuracy card with a delta showing the baseline for comparison
    "ML Accuracy",
    f"{metrics['ml_accuracy']:.1%}" if metrics else "—",
    delta=f"baseline {metrics['baseline_accuracy']:.1%}" if metrics else None,  # shows what the dumb always-predict-majority baseline achieves
)

if metrics:  # only show the buy-and-hold comparison banner if the pipeline has run
    bnh = metrics["buy_and_hold_return"]  # simple buy-and-hold return over the same period
    strat_ret = metrics["final_balance"] / INITIAL_BALANCE - 1  # BUG FIX: uses INITIAL_BALANCE from config; strategy return = (final - start) / start
    colour = "green" if strat_ret >= bnh else "red"  # green if strategy beat buy-and-hold, red if it underperformed
    diff = strat_ret - bnh  # how much the strategy outperformed (positive) or underperformed (negative) buy-and-hold
    st.markdown(  # st.markdown(): render a string of Markdown/HTML in the page; unsafe_allow_html=True is required to use the <span> colour tag
        f"**Buy-and-hold benchmark:** {bnh:.1%} &nbsp;|&nbsp; "  # bold label with the buy-and-hold return; &nbsp; = non-breaking space for visual padding
        f"Strategy outperforms by <span style='color:{colour}'>{diff:+.1%}</span>",  # coloured span: + in format string forces a + or - sign to always show
        unsafe_allow_html=True,  # unsafe_allow_html=True: allow raw HTML tags in the markdown — needed for the <span style=...> colour tag
    )

left, right = st.columns([2, 1])  # st.columns([2,1]): split into two columns where the left is 2× wider than the right
with left:  # context manager: UI elements inside this block appear in the left column
    st.subheader("Backtest Chart")  # st.subheader(): H3-level heading
    if chart_path.exists():  # check if the chart image file exists on disk
        st.image(str(chart_path), use_container_width=True)  # st.image(): display the PNG image; use_container_width=True: scale it to fill the column width
    else:
        st.info("Run the pipeline to generate dashboard.png.")  # st.info(): blue info box with a hint

with right:  # UI elements here appear in the narrower right column
    st.subheader("Latest Trade Log")
    if trade_log_df.empty:  # .empty: True if the DataFrame has no rows (file not found or no trades)
        st.info("trade_log.csv not found yet.")
    else:
        st.dataframe(trade_log_df.tail(10), use_container_width=True)  # st.dataframe(): render an interactive scrollable table; .tail(10): show only the last 10 trades

st.subheader("Machine Learning Snapshot")  # section heading for the ML results
if ml_report_df.empty:  # no ML report saved yet
    st.info("ml_report.csv not found yet.")
else:
    metrics_section = ml_report_df[ml_report_df["section"] == "metrics"]  # boolean indexing: keep only rows where the "section" column equals "metrics"
    feature_section = ml_report_df[ml_report_df["section"] == "feature_importance"]  # keep only feature importance rows
    if not metrics_section.empty:  # only show if there are actual rows
        st.dataframe(metrics_section, use_container_width=True, hide_index=True)  # render the metrics table; hide_index=True: don't show the row numbers
    if not feature_section.empty:
        st.bar_chart(feature_section.set_index("name")["value"])  # st.bar_chart(): render a bar chart; .set_index("name"): make feature names the x-axis labels; ["value"]: use the weight column as bar heights

st.markdown(  # st.markdown(): render the code block below as formatted Markdown
    """
Run locally with:

```bash
streamlit run app.py
```
"""
)
