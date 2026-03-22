from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from quant_bot.config import DEFAULT_DAYS, DEFAULT_INTERVAL, DEFAULT_ML_HORIZON, DEFAULT_SYMBOL
from quant_bot.pipeline import run_full_pipeline


st.set_page_config(page_title="BTC Quant Dashboard", page_icon="BTC", layout="wide")


def load_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


st.title("BTC Quant Trading Bot")
st.caption("Backtest analytics plus a first-pass machine learning experiment.")

with st.sidebar:
    st.header("Run Settings")
    symbol = st.text_input("Symbol", value=DEFAULT_SYMBOL)
    interval = st.selectbox("Interval", options=["1h", "4h", "1d"], index=0)
    days = st.slider("Lookback days", min_value=90, max_value=730, value=DEFAULT_DAYS, step=30)
    ml_horizon = st.slider("ML horizon (candles ahead)", min_value=3, max_value=48, value=DEFAULT_ML_HORIZON)
    run_button = st.button("Run fresh analysis", type="primary")

if run_button:
    with st.spinner("Fetching market data and running the pipeline..."):
        df, result, ml_result = run_full_pipeline(
            symbol=symbol,
            interval=interval,
            days=days,
            ml_horizon=ml_horizon,
        )
    st.success("Analysis complete.")
    st.session_state["latest_metrics"] = {
        "candles": len(df),
        "final_balance": result.final_balance,
        "max_drawdown": result.max_drawdown,
        "ml_accuracy": ml_result.metrics["accuracy"],
        "baseline_accuracy": ml_result.metrics["baseline_accuracy"],
    }

trade_log_df = load_csv("trade_log.csv")
ml_report_df = load_csv("ml_report.csv")
chart_path = Path("dashboard.png")

metrics = st.session_state.get("latest_metrics", {})
col1, col2, col3, col4 = st.columns(4)
col1.metric("Candles", f"{int(metrics.get('candles', 0)):,}" if metrics else "Saved run")
col2.metric("Final Balance", f"${metrics.get('final_balance', 0):,.2f}" if metrics else "Open trade log")
col3.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.1%}" if metrics else "Open chart")
col4.metric("ML Accuracy", f"{metrics.get('ml_accuracy', 0):.1%}" if metrics else "Open ML report")

left, right = st.columns([2, 1])
with left:
    st.subheader("Backtest Chart")
    if chart_path.exists():
        st.image(str(chart_path), use_container_width=True)
    else:
        st.info("Run the pipeline once to generate dashboard.png.")

with right:
    st.subheader("Latest Trade Log")
    if trade_log_df.empty:
        st.info("trade_log.csv not found yet.")
    else:
        st.dataframe(trade_log_df.tail(10), use_container_width=True)

st.subheader("Machine Learning Snapshot")
if ml_report_df.empty:
    st.info("ml_report.csv not found yet.")
else:
    metrics_df = ml_report_df[ml_report_df["section"] == "metrics"]
    feature_df = ml_report_df[ml_report_df["section"] == "feature_importance"]
    if not metrics_df.empty:
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    if not feature_df.empty:
        st.bar_chart(feature_df.set_index("name")["value"])

st.markdown(
    """
Run locally with:

```bash
streamlit run app.py
```
"""
)
