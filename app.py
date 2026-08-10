from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from quant_bot.config import (
    DEFAULT_DAYS,
    DEFAULT_INTERVAL,
    DEFAULT_ML_HORIZON,
    DEFAULT_SYMBOL,
    INITIAL_BALANCE,
    LIVE_STATE_PATH,
    NOTABLE_PERIODS,
)
from quant_bot.pipeline import run_full_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

st.set_page_config(page_title="BTC Quant Dashboard", page_icon="📈", layout="wide")


def load_csv(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def load_live_state() -> dict:
    p = Path(LIVE_STATE_PATH)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="Training XGBoost model on latest data...")
def get_trained_model():
    """Fetch latest data and train the ML model once per Streamlit session."""
    from quant_bot.data import get_price_data
    from quant_bot.ml import run_ml_experiment
    df = get_price_data(symbol=DEFAULT_SYMBOL, interval=DEFAULT_INTERVAL, days=DEFAULT_DAYS)
    result = run_ml_experiment(df, horizon=DEFAULT_ML_HORIZON)
    return result.model


@st.cache_resource(show_spinner="Connecting to Binance testnet...")
def get_binance_client(api_key: str, api_secret: str, testnet: bool = True):
    """Create the Binance REST client. Cached across reruns so we do not reconnect constantly."""
    from quant_bot.binance_client import BinanceClient
    return BinanceClient(api_key=api_key, api_secret=api_secret, testnet=testnet)


st.title("BTC Quant Trading Bot")
st.caption("Backtest historical performance and paper-trade on Binance testnet from the same dashboard.")


tab_backtest, tab_paper = st.tabs(["📊 Backtest", "🤖 Paper Trade"])


with tab_backtest:
    with st.sidebar:
        st.header("Backtest Settings")
        symbol = st.text_input("Symbol", value=DEFAULT_SYMBOL, key="bt_symbol")
        interval = st.selectbox("Interval", options=["1h", "4h", "1d"], index=0, key="bt_interval")
        days = st.slider("Lookback days", min_value=90, max_value=1095, value=DEFAULT_DAYS, step=30, key="bt_days")

        st.markdown("---")
        st.subheader("Historical Period")
        st.caption("Leave end date blank to use today. Set it to test a past window.")

        preset_options = ["(none — use end date below)"] + list(NOTABLE_PERIODS.keys())
        selected_preset = st.selectbox(
            "Quick preset",
            options=preset_options,
            index=0,
            help="Presets automatically set days + end date for well-known market periods",
            key="bt_preset",
        )

        if selected_preset != preset_options[0]:
            preset_vals = NOTABLE_PERIODS[selected_preset]
            days = preset_vals["days"]
            preset_end = preset_vals["end_date"]
            st.text_input("End date (YYYY-MM-DD)", value=preset_end if preset_end else "", disabled=True, key="bt_enddate_readonly")
            end_date = preset_end
        else:
            end_date_input = st.text_input(
                "End date (YYYY-MM-DD)",
                value="",
                placeholder="e.g. 2021-11-10",
                help="Leave blank for the most recent data. Enter a past date to analyse a historical window.",
                key="bt_enddate",
            )
            end_date = end_date_input.strip() if end_date_input.strip() else None

        st.markdown("---")
        ml_horizon = st.slider("ML horizon (candles ahead)", min_value=3, max_value=48, value=DEFAULT_ML_HORIZON, key="bt_horizon")
        run_button = st.button("Run fresh analysis", type="primary", key="bt_run")

    if run_button:
        with st.spinner("Fetching market data, training ML model, running backtest…"):
            df, result, ml_result = run_full_pipeline(
                symbol=symbol,
                interval=interval,
                days=days,
                end_date=end_date,
                ml_horizon=ml_horizon,
            )
        st.success("Analysis complete.")
        st.session_state["latest_metrics"] = {
            "candles":             len(df),
            "final_balance":       result.final_balance,
            "max_drawdown":        result.max_drawdown,
            "sharpe_ratio":        result.sharpe_ratio,
            "buy_and_hold_return": result.buy_and_hold_return,
            "ml_accuracy":         ml_result.metrics["accuracy"],
            "baseline_accuracy":   ml_result.metrics["baseline_accuracy"],
        }

    trade_log_df = load_csv("trade_log.csv")
    ml_report_df = load_csv("ml_report.csv")
    chart_path = Path("dashboard.png")
    metrics = st.session_state.get("latest_metrics", {})

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Candles", f"{int(metrics['candles']):,}" if metrics else "—")
    col2.metric(
        "Final Balance",
        f"${metrics['final_balance']:,.2f}" if metrics else "—",
        delta=f"{(metrics['final_balance'] / INITIAL_BALANCE - 1):.1%} vs start" if metrics else None,
    )
    col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}" if metrics else "—")
    col4.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}" if metrics else "—")
    col5.metric(
        "ML Accuracy",
        f"{metrics['ml_accuracy']:.1%}" if metrics else "—",
        delta=f"baseline {metrics['baseline_accuracy']:.1%}" if metrics else None,
    )

    if metrics:
        bnh = metrics["buy_and_hold_return"]
        strat_ret = metrics["final_balance"] / INITIAL_BALANCE - 1
        colour = "green" if strat_ret >= bnh else "red"
        diff = strat_ret - bnh
        st.markdown(
            f"**Buy-and-hold benchmark:** {bnh:.1%} &nbsp;|&nbsp; "
            f"Strategy outperforms by <span style='color:{colour}'>{diff:+.1%}</span>",
            unsafe_allow_html=True,
        )

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Backtest Chart")
        if chart_path.exists():
            st.image(str(chart_path), use_container_width=True)
        else:
            st.info("Run the pipeline to generate dashboard.png.")

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
        metrics_section = ml_report_df[ml_report_df["section"] == "metrics"]
        feature_section = ml_report_df[ml_report_df["section"] == "feature_importance"]
        if not metrics_section.empty:
            st.dataframe(metrics_section, use_container_width=True, hide_index=True)
        if not feature_section.empty:
            st.bar_chart(feature_section.set_index("name")["value"])


with tab_paper:
    st.header("Paper Trading on Binance Testnet")
    st.caption(
        "Trade with **fake money on real-time prices** — perfect for validating the bot "
        "before risking real funds. Get free testnet API keys at "
        "[testnet.binance.vision](https://testnet.binance.vision/)."
    )

    env_key    = os.environ.get("BINANCE_API_KEY", "").strip()
    env_secret = os.environ.get("BINANCE_API_SECRET", "").strip()

    with st.expander("🔑 API credentials", expanded=(not env_key)):
        st.caption("Keys are stored only in this session's memory — never written to disk.")
        if env_key and env_secret:
            st.success("✓ Credentials loaded from environment variables (BINANCE_API_KEY / BINANCE_API_SECRET).")
            api_key    = env_key
            api_secret = env_secret
        else:
            api_key = st.text_input(
                "API Key",
                value=st.session_state.get("pt_api_key", ""),
                type="password",
                key="pt_api_key_input",
            )
            api_secret = st.text_input(
                "API Secret",
                value=st.session_state.get("pt_api_secret", ""),
                type="password",
                key="pt_api_secret_input",
            )
            st.session_state["pt_api_key"] = api_key
            st.session_state["pt_api_secret"] = api_secret

    if not api_key or not api_secret:
        st.warning("⚠️ Enter your Binance testnet API key and secret above to enable paper trading.")
        st.stop()

    try:
        client = get_binance_client(api_key, api_secret, testnet=True)
        usdt_balance = client.get_balance("USDT")
        btc_balance  = client.get_balance("BTC")
        connected = True
    except Exception as exc:
        st.error(f"❌ Connection failed: {exc}")
        st.caption("Check that your API keys are for the **testnet** (not mainnet) and are still valid.")
        connected = False
        usdt_balance = btc_balance = 0.0

    if not connected:
        st.stop()

    try:
        latest_kline = client.get_klines(symbol=DEFAULT_SYMBOL, interval=DEFAULT_INTERVAL, limit=1)
        current_price = float(latest_kline[-1][4])
    except Exception:
        current_price = 0.0

    portfolio_value = usdt_balance + btc_balance * current_price

    status_col, price_col = st.columns([1, 2])
    with status_col:
        st.success(f"✅ Connected to Binance **Testnet**")
    with price_col:
        st.info(f"📈 Current BTC price: **${current_price:,.2f}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("USDT Balance", f"${usdt_balance:,.2f}")
    c2.metric("BTC Balance", f"{btc_balance:.6f} BTC", delta=f"${btc_balance * current_price:,.2f}" if btc_balance > 0 else None)
    c3.metric("Portfolio Value", f"${portfolio_value:,.2f}")

    live_state = load_live_state()
    st.subheader("Current Position")

    if live_state.get("in_trade", False):
        pos_col1, pos_col2, pos_col3 = st.columns(3)
        entry_price = live_state.get("buy_price", 0.0)
        peak_price = live_state.get("peak_price", 0.0)
        pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price else 0
        pos_col1.metric("Status", "🟢 IN TRADE")
        pos_col2.metric("Entry Price", f"${entry_price:,.2f}", delta=f"{pnl_pct:+.2f}% unrealised")
        pos_col3.metric("Trailing Stop", f"{live_state.get('current_trailing_stop', 0.05):.0%}")
        drop_from_peak = ((peak_price - current_price) / peak_price * 100) if peak_price else 0
        st.caption(f"Peak since entry: ${peak_price:,.2f} | Current drop from peak: {drop_from_peak:.2f}%")
    else:
        st.info("💰 **Status: In cash** — waiting for the next entry signal.")

    st.subheader("Actions")
    act_col1, act_col2, act_col3 = st.columns(3)

    with act_col1:
        if st.button("🎯 Trade Once", type="primary", help="Run one decision cycle: check signal, place order if conditions are met"):
            with st.spinner("Fetching data, training model (if needed), checking signal…"):
                try:
                    model = get_trained_model()
                    from quant_bot.live_trader import trade_once
                    new_state = trade_once(client, model)
                    st.success("Trade cycle complete — see updated status above.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Trade cycle failed: {exc}")

    with act_col2:
        if st.button("🔄 Refresh Balances", help="Poll Binance for latest balances and price"):
            st.rerun()

    with act_col3:
        if st.button("🧹 Reset State File", help="Clear live_state.json (does NOT close any real position on Binance)"):
            p = Path(LIVE_STATE_PATH)
            if p.exists():
                p.unlink()
                st.success("State file cleared.")
                st.rerun()
            else:
                st.info("No state file to clear.")

    st.subheader("Recent Trades")
    history = live_state.get("trade_history", [])
    if not history:
        st.info("No trades yet. Click **Trade Once** to check for a signal.")
    else:
        trades_df = pd.DataFrame(history)
        st.dataframe(trades_df.tail(20), use_container_width=True, hide_index=True)

        if len(trades_df) >= 2 and "pnl_usdt" in trades_df.columns:
            st.subheader("Equity Curve (from trade history)")
            trades_df["cumulative_pnl"] = trades_df["pnl_usdt"].cumsum()
            trades_df["equity"] = INITIAL_BALANCE + trades_df["cumulative_pnl"]
            equity_series = trades_df.set_index("exit_time")["equity"]
            st.line_chart(equity_series)

    with st.expander("⏰ How to automate hourly checks"):
        st.markdown(
            """
            **Option A — Windows Task Scheduler (recommended):**
            1. Open Task Scheduler → Create Basic Task
            2. Trigger: Daily, repeat every 1 hour indefinitely
            3. Action: `python.exe`
            4. Arguments: `live.py`
            5. Start in: `C:\\Users\\nello\\OneDrive\\Desktop\\IMPORTANT\\quant`

            **Option B — Terminal loop mode:**
            ```bash
            python live.py loop
            ```
            Keeps a terminal window open; checks every hour. Ctrl+C to stop.

            **Option C — Use this dashboard manually:**
            Click 🎯 **Trade Once** whenever you want the bot to check for a signal.
            Good for learning / debugging, but you must be at your computer.
            """
        )

    st.caption("⚠️ Currently connected to **testnet** (fake money). To go live with real money, edit `get_binance_client(..., testnet=False)` — but only after validating on testnet for weeks.")
