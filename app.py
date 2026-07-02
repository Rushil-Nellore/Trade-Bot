from __future__ import annotations  # enables newer type-hint syntax on Python 3.9 and earlier

import json     # standard library — used to read/write live_state.json for the paper-trade tab
import logging  # Python's built-in logging library — forward logs to Streamlit's terminal
import os       # standard library — read Binance API credentials from environment variables
from pathlib import Path  # object-oriented file path handling — cleaner than string manipulation

import pandas as pd  # data-table library — used to load and display CSVs
import streamlit as st  # streamlit: the web dashboard framework — every st.* call renders a UI element

from quant_bot.config import (
    DEFAULT_DAYS,
    DEFAULT_INTERVAL,
    DEFAULT_ML_HORIZON,
    DEFAULT_SYMBOL,
    INITIAL_BALANCE,
    LIVE_STATE_PATH,          # NEW: filename where live_trader.py persists cash/position between runs
    NOTABLE_PERIODS,
)
from quant_bot.pipeline import run_full_pipeline  # runs the backtest pipeline (fetch data → train ML → simulate → save outputs)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

st.set_page_config(page_title="BTC Quant Dashboard", page_icon="📈", layout="wide")  # configure the browser tab


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:  # safely load a CSV — returns empty DataFrame if the file does not exist
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_csv(file_path)


def load_live_state() -> dict:  # load the live-trader's state file (or empty dict if not yet created)
    p = Path(LIVE_STATE_PATH)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


# ── Cache the heavy resources so we do not retrain / reconnect on every rerun ──
# Streamlit reruns the entire script on every widget change; caching prevents
# retraining the XGBoost model and re-creating the Binance client each time.

@st.cache_resource(show_spinner="Training XGBoost model on latest data...")  # @st.cache_resource: expensive object stored across reruns
def get_trained_model():
    """Fetch latest data and train the ML model once per Streamlit session."""
    from quant_bot.data import get_price_data  # imported inside function so the app loads even if network is down
    from quant_bot.ml import run_ml_experiment
    df = get_price_data(symbol=DEFAULT_SYMBOL, interval=DEFAULT_INTERVAL, days=DEFAULT_DAYS)
    result = run_ml_experiment(df, horizon=DEFAULT_ML_HORIZON)
    return result.model


@st.cache_resource(show_spinner="Connecting to Binance testnet...")
def get_binance_client(api_key: str, api_secret: str, testnet: bool = True):
    """Create the Binance REST client. Cached across reruns so we do not reconnect constantly."""
    from quant_bot.binance_client import BinanceClient
    return BinanceClient(api_key=api_key, api_secret=api_secret, testnet=testnet)


# ── Title bar ─────────────────────────────────────────────────────────────────

st.title("BTC Quant Trading Bot")
st.caption("Backtest historical performance and paper-trade on Binance testnet from the same dashboard.")


# ── Tabs — Backtest and Paper Trade sit side-by-side ─────────────────────────

tab_backtest, tab_paper = st.tabs(["📊 Backtest", "🤖 Paper Trade"])  # st.tabs(): renders clickable tab bar; returns one context manager per tab


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — BACKTEST (original functionality, unchanged behaviour)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_backtest:  # everything indented under this block renders inside the Backtest tab
    with st.sidebar:  # sidebar is shared across tabs — put backtest controls here
        st.header("Backtest Settings")
        symbol = st.text_input("Symbol", value=DEFAULT_SYMBOL, key="bt_symbol")  # key= disambiguates from any similarly-named widget in the other tab
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

        if selected_preset != preset_options[0]:  # user picked a preset
            preset_vals = NOTABLE_PERIODS[selected_preset]
            days = preset_vals["days"]
            preset_end = preset_vals["end_date"]
            st.text_input("End date (YYYY-MM-DD)", value=preset_end if preset_end else "", disabled=True, key="bt_enddate_readonly")
            end_date = preset_end
        else:  # custom / no preset
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

    # ── Run backtest on button click ──────────────────────────────────────────
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
        st.session_state["latest_metrics"] = {  # persist across reruns
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

    # ── Five metric cards ────────────────────────────────────────────────────
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

    # ── Buy-and-hold benchmark banner ────────────────────────────────────────
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

    # ── Chart + trade log side by side ───────────────────────────────────────
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

    # ── ML snapshot ──────────────────────────────────────────────────────────
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


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PAPER TRADE (NEW — live trading against Binance testnet)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_paper:  # everything indented under this block renders inside the Paper Trade tab
    st.header("Paper Trading on Binance Testnet")
    st.caption(
        "Trade with **fake money on real-time prices** — perfect for validating the bot "
        "before risking real funds. Get free testnet API keys at "
        "[testnet.binance.vision](https://testnet.binance.vision/)."
    )

    # ── Step 1 — Credentials ──────────────────────────────────────────────────
    # Prefer environment variables; fall back to UI inputs stored in session_state.
    # Session state means the keys persist during THIS browser session but are
    # never written to disk by Streamlit.
    env_key    = os.environ.get("BINANCE_API_KEY", "").strip()
    env_secret = os.environ.get("BINANCE_API_SECRET", "").strip()

    with st.expander("🔑 API credentials", expanded=(not env_key)):  # expander: collapsible box; open by default if creds missing
        st.caption("Keys are stored only in this session's memory — never written to disk.")
        if env_key and env_secret:
            st.success("✓ Credentials loaded from environment variables (BINANCE_API_KEY / BINANCE_API_SECRET).")
            api_key    = env_key
            api_secret = env_secret
        else:
            api_key = st.text_input(
                "API Key",
                value=st.session_state.get("pt_api_key", ""),
                type="password",  # type="password": characters shown as bullets
                key="pt_api_key_input",
            )
            api_secret = st.text_input(
                "API Secret",
                value=st.session_state.get("pt_api_secret", ""),
                type="password",
                key="pt_api_secret_input",
            )
            # Persist across reruns in session_state (in-memory only)
            st.session_state["pt_api_key"] = api_key
            st.session_state["pt_api_secret"] = api_secret

    # ── Step 2 — Connection status ────────────────────────────────────────────
    if not api_key or not api_secret:
        st.warning("⚠️ Enter your Binance testnet API key and secret above to enable paper trading.")
        st.stop()  # st.stop(): halt rendering of the rest of this tab — user must complete setup first

    # ── Try to connect and fetch balances ────────────────────────────────────
    try:
        client = get_binance_client(api_key, api_secret, testnet=True)  # cached — only builds once per session
        usdt_balance = client.get_balance("USDT")  # live query — returns current testnet USDT
        btc_balance  = client.get_balance("BTC")   # live query — returns current testnet BTC
        connected = True
    except Exception as exc:  # any connection or auth error
        st.error(f"❌ Connection failed: {exc}")
        st.caption("Check that your API keys are for the **testnet** (not mainnet) and are still valid.")
        connected = False
        usdt_balance = btc_balance = 0.0

    if not connected:
        st.stop()  # cannot proceed without a working connection

    # ── Step 3 — Fetch latest BTC price for portfolio valuation ──────────────
    try:
        latest_kline = client.get_klines(symbol=DEFAULT_SYMBOL, interval=DEFAULT_INTERVAL, limit=1)  # get most recent hourly candle
        current_price = float(latest_kline[-1][4])  # index [4] in a kline row is the close price
    except Exception:
        current_price = 0.0

    portfolio_value = usdt_balance + btc_balance * current_price  # total account value in USD-equivalent

    # ── Step 4 — Connection status + balance cards ───────────────────────────
    status_col, price_col = st.columns([1, 2])
    with status_col:
        st.success(f"✅ Connected to Binance **Testnet**")
    with price_col:
        st.info(f"📈 Current BTC price: **${current_price:,.2f}**")

    c1, c2, c3 = st.columns(3)
    c1.metric("USDT Balance", f"${usdt_balance:,.2f}")
    c2.metric("BTC Balance", f"{btc_balance:.6f} BTC", delta=f"${btc_balance * current_price:,.2f}" if btc_balance > 0 else None)
    c3.metric("Portfolio Value", f"${portfolio_value:,.2f}")

    # ── Step 5 — Current position status (from live_state.json) ──────────────
    live_state = load_live_state()  # read the persisted state file written by live_trader.py
    st.subheader("Current Position")

    if live_state.get("in_trade", False):  # bot is currently holding BTC
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

    # ── Step 6 — Action buttons ──────────────────────────────────────────────
    st.subheader("Actions")
    act_col1, act_col2, act_col3 = st.columns(3)

    with act_col1:
        if st.button("🎯 Trade Once", type="primary", help="Run one decision cycle: check signal, place order if conditions are met"):
            with st.spinner("Fetching data, training model (if needed), checking signal…"):
                try:
                    model = get_trained_model()  # cached across reruns — retrains only on first call this session
                    from quant_bot.live_trader import trade_once
                    new_state = trade_once(client, model)
                    st.success("Trade cycle complete — see updated status above.")
                    st.rerun()  # st.rerun(): force the page to re-execute so balances/state refresh
                except Exception as exc:
                    st.error(f"Trade cycle failed: {exc}")

    with act_col2:
        if st.button("🔄 Refresh Balances", help="Poll Binance for latest balances and price"):
            # Clear cached balances so they get re-fetched — st.cache_resource caches the client, not the balance calls
            st.rerun()

    with act_col3:
        if st.button("🧹 Reset State File", help="Clear live_state.json (does NOT close any real position on Binance)"):
            p = Path(LIVE_STATE_PATH)
            if p.exists():
                p.unlink()  # delete the state file
                st.success("State file cleared.")
                st.rerun()
            else:
                st.info("No state file to clear.")

    # ── Step 7 — Trade history table ─────────────────────────────────────────
    st.subheader("Recent Trades")
    history = live_state.get("trade_history", [])  # list of trade dicts written by live_trader.py
    if not history:
        st.info("No trades yet. Click **Trade Once** to check for a signal.")
    else:
        trades_df = pd.DataFrame(history)  # convert list of dicts to a DataFrame
        st.dataframe(trades_df.tail(20), use_container_width=True, hide_index=True)  # show last 20 trades

        # ── Step 8 — Equity curve from trade history ─────────────────────────
        if len(trades_df) >= 2 and "pnl_usdt" in trades_df.columns:
            st.subheader("Equity Curve (from trade history)")
            trades_df["cumulative_pnl"] = trades_df["pnl_usdt"].cumsum()  # cumsum(): running total of P&L
            trades_df["equity"] = INITIAL_BALANCE + trades_df["cumulative_pnl"]  # starting balance + cumulative P&L
            equity_series = trades_df.set_index("exit_time")["equity"]  # x-axis = exit_time, y-axis = equity
            st.line_chart(equity_series)  # st.line_chart(): interactive line chart

    # ── Step 9 — How to automate the hourly loop ─────────────────────────────
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
