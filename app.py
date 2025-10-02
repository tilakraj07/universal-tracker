# ==============================================
# UNIVERSAL TRACKER — Portfolio + Intraday + Alerts + Telegram (Hardcoded)
# ==============================================
# HOW TO RUN LOCALLY:
# 1) Save this as app.py
# 2) pip install streamlit streamlit-autorefresh yfinance pandas numpy requests
# 3) python -m streamlit run app.py
#
# HOW TO DEPLOY (Streamlit Cloud):
# - Push app.py + requirements.txt to GitHub
# - Deploy at https://share.streamlit.io
# ==============================================

import os
import hashlib
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import requests
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Universal Tracker — Portfolio + Intraday + Alerts", layout="wide")
st.title("📊 Universal Tracker — Portfolio + Intraday + Alerts (Telegram Enabled)")

# ---- Auto-refresh every 15 minutes ----
st_autorefresh(interval=900_000, key="auto15min")

PORTFOLIO_CSV = "portfolio.csv"

# ---------------------------------
# Telegram setup (HARDCODED)
# ---------------------------------
tg_enable = True
tg_token  = "8298370446:AAHQJdZpq1TZumNG3tacLpBnH6Ge6cCJU3o"
tg_chat   = "888880398"

def send_telegram(msg: str):
    if not (tg_enable and tg_token and tg_chat):
        return
    try:
        url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
        requests.post(url, json={"chat_id": tg_chat, "text": msg})
    except Exception:
        pass  # don't crash if Telegram fails

# ---------------------------------
# Helpers & classification
# ---------------------------------
def asset_kind(sym: str) -> str:
    s = sym.upper()
    if s.endswith(".NS") or s.endswith(".BO"):
        return "stock"
    if s.endswith("-USD"):
        return "crypto"
    if s in {"GC=F", "SI=F"} or s.endswith("=F"):
        return "futures"
    if s.endswith("=X"):
        return "fx"
    return "other"

# ---------------------------------
# CACHED FETCHERS
# ---------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_intraday_general(ticker: str, period="60d", interval="15m") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60, show_spinner=False)
def fetch_intraday_crypto(ticker: str, period="7d", interval="1m") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900, show_spinner=False)
def fetch_daily(ticker: str, period="90d") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=False)
        if df is None or df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except Exception:
        return pd.DataFrame()

# ---------------------------------
# Indicators
# ---------------------------------
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def macd(series: pd.Series):
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist

def macd_cross_state(close: pd.Series) -> tuple[str, float, float, float]:
    macd_line, signal, hist = macd(close)
    try:
        curr_macd = float(macd_line.iloc[-1])
        prev_macd = float(macd_line.iloc[-2])
        curr_sig  = float(signal.iloc[-1])
        prev_sig  = float(signal.iloc[-2])
    except Exception:
        return "None", 0.0, 0.0, 0.0
    bull = (curr_macd > curr_sig) and (prev_macd <= prev_sig)
    bear = (curr_macd < curr_sig) and (prev_macd >= prev_sig)
    cross_state = "Bullish" if bull else ("Bearish" if bear else "None")
    return cross_state, curr_macd, curr_sig, float(hist.iloc[-1])

# ---------------------------------
# Defaults & fallback
# ---------------------------------
DEFAULT_SYMBOLS = [
    "RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS",
    "BTC-USD", "ETH-USD", "SOL-USD",
    "XAUUSD=X", "XAGUSD=X", "GC=F", "SI=F",
]
FALLBACK = {"XAUUSD=X": "GC=F", "XAGUSD=X": "SI=F"}

st.caption("Intraday: Stocks/Metals 15m, Crypto 1m. Auto-refresh every 15 min. Telegram alerts enabled.")

# ---------------------------------
# Portfolio persistence
# ---------------------------------
def load_or_init_portfolio() -> pd.DataFrame:
    if os.path.exists(PORTFOLIO_CSV):
        try:
            df = pd.read_csv(PORTFOLIO_CSV)
            for col in ["Symbol","Quantity","Purchase Price","Stop Level","Target Price","Notes"]:
                if col not in df.columns:
                    df[col] = np.nan if col not in ["Notes"] else ""
            return df[["Symbol","Quantity","Purchase Price","Stop Level","Target Price","Notes"]]
        except Exception:
            pass
    return pd.DataFrame({
        "Symbol": DEFAULT_SYMBOLS,
        "Quantity": np.nan,
        "Purchase Price": np.nan,
        "Stop Level": np.nan,
        "Target Price": np.nan,
        "Notes": "",
    })

if "portfolio" not in st.session_state:
    st.session_state.portfolio = load_or_init_portfolio()

if "sent_alerts" not in st.session_state:
    st.session_state.sent_alerts = set()

st.sidebar.markdown(f"**Last refreshed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ---------------------------------
# Portfolio editor
# ---------------------------------
st.subheader("Edit your portfolio")
edited = st.data_editor(
    st.session_state.portfolio,
    num_rows="dynamic",
    use_container_width=True,
    key="portfolio_editor",
    column_config={
        "Quantity": st.column_config.NumberColumn(format="%.4f", step=1.0),
        "Purchase Price": st.column_config.NumberColumn(format="%.4f", step=0.01),
        "Stop Level": st.column_config.NumberColumn(format="%.4f", step=0.01),
        "Target Price": st.column_config.NumberColumn(format="%.4f", step=0.01),
        "Notes": st.column_config.TextColumn(),
    }
)
st.session_state.portfolio = edited.copy()

# Autosave
def df_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df.fillna(""), index=False).values).hexdigest()
if "last_save_hash" not in st.session_state:
    st.session_state.last_save_hash = ""
cur_hash = df_hash(st.session_state.portfolio)
if cur_hash != st.session_state.last_save_hash:
    try:
        st.session_state.portfolio.to_csv(PORTFOLIO_CSV, index=False)
        st.session_state.last_save_hash = cur_hash
        st.sidebar.success("Autosaved ✅", icon="💾")
    except Exception as e:
        st.sidebar.error(f"Autosave failed: {e}")

# ---------------------------------
# Build table & alerts
# ---------------------------------
rows, failed, alert_items = [], [], []

with st.spinner("Fetching intraday & daily…"):
    for sym in edited["Symbol"].dropna().astype(str):
        kind = asset_kind(sym)
        if kind == "crypto":
            df = fetch_intraday_crypto(sym, period="7d", interval="1m")
        else:
            df = fetch_intraday_general(sym, period="60d", interval="15m")
            if (df.empty or len(df) < 210) and sym in FALLBACK:
                df = fetch_intraday_general(FALLBACK[sym], period="60d", interval="15m")

        df_daily = fetch_daily(sym, period="90d")
        if df.empty or "Close" not in df.columns:
            failed.append(sym)
            continue

        close = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if len(close) < 200:
            failed.append(sym)
            continue

        last_price = float(close.iloc[-1])
        ema200 = ema(close, 200)
        cross_state, macd_val, macd_sig, macd_hist = macd_cross_state(close)

        # Daily % changes
        pct2 = pct5 = pct7 = None
        if not df_daily.empty and "Close" in df_daily.columns and len(df_daily) >= 8:
            dclose = pd.to_numeric(df_daily["Close"], errors="coerce").dropna()
            if len(dclose) >= 3:
                pct2 = (dclose.iloc[-1] / dclose.shift(2).iloc[-1] - 1.0) * 100.0
            if len(dclose) >= 6:
                pct5 = (dclose.iloc[-1] / dclose.shift(5).iloc[-1] - 1.0) * 100.0
            if len(dclose) >= 8:
                pct7 = (dclose.iloc[-1] / dclose.shift(7).iloc[-1] - 1.0) * 100.0

        # Portfolio values
        row_user = edited[edited["Symbol"] == sym].iloc[0]
        qty    = row_user.get("Quantity", np.nan)
        buy    = row_user.get("Purchase Price", np.nan)
        stop   = row_user.get("Stop Level", np.nan)
        target = row_user.get("Target Price", np.nan)
        notes  = row_user.get("Notes", "")

        pl_pct = (last_price / float(buy) - 1.0) * 100.0 if pd.notna(buy) and buy != 0 else None
        pl_amt = (last_price - float(buy)) * float(qty) if pd.notna(buy) and pd.notna(qty) else None
        below_stop = (last_price <= float(stop)) if pd.notna(stop) and stop != 0 else None
        above_target = (last_price >= float(target)) if pd.notna(target) and target != 0 else None

        # Alerts
        def push_alert(key: str, message: str):
            if key not in st.session_state.sent_alerts:
                st.session_state.sent_alerts.add(key)
                alert_items.append(message)
                send_telegram(message)

        # 1) P/L ±5% or ±10%
        if pl_pct is not None:
            if pl_pct >= 10: push_alert(f"{sym}:PL+10", f"🔔 {sym}: P/L +{pl_pct:.2f}% (≥10%)")
            elif pl_pct <= -10: push_alert(f"{sym}:PL-10", f"🔔 {sym}: P/L {pl_pct:.2f}% (≤-10%)")
            if 5 <= pl_pct < 10: push_alert(f"{sym}:PL+5", f"🔔 {sym}: P/L +{pl_pct:.2f}% (≥5%)")
            elif -10 < pl_pct <= -5: push_alert(f"{sym}:PL-5", f"🔔 {sym}: P/L {pl_pct:.2f}% (≤-5%)")

        # 2) 2-day change ±3%
        if pct2 is not None:
            if pct2 >= 3: push_alert(f"{sym}:2D+3", f"📈 {sym}: +{pct2:.2f}% in 2 days")
            elif pct2 <= -3: push_alert(f"{sym}:2D-3", f"📉 {sym}: {pct2:.2f}% in 2 days")

        # 3) Stop level
        if below_stop is True:
            push_alert(f"{sym}:STOP", f"⛔ {sym}: {last_price:.2f} ≤ Stop {float(stop):.2f}")

        # 4) Target price
        if above_target is True:
            push_alert(f"{sym}:TARGET", f"🎯 {sym}: {last_price:.2f} ≥ Target {float(target):.2f}")

        # Symbol markers
        symbol_label = sym
        if pl_pct is not None and pl_pct <= -5: symbol_label = "⛔ " + symbol_label
        elif pct7 is not None and pct7 >= 3: symbol_label = "⬆️ " + symbol_label

        rows.append({
            "Symbol": symbol_label,
            "Date/Time": close.index[-1],
            "Current Price": round(last_price, 4),
            "2D %": None if pct2 is None else round(pct2, 2),
            "5D %": None if pct5 is None else round(pct5, 2),
            "7D %": None if pct7 is None else round(pct7, 2),
            "200 EMA (intraday)": round(float(ema200.iloc[-1]), 4),
            "Price vs 200 EMA": "Above" if last_price > float(ema200.iloc[-1]) else "Below",
            "MACD": round(float(macd_val), 5),
            "MACD Signal": round(float(macd_sig), 5),
            "MACD Hist": round(float(macd_hist), 5),
            "MACD Crossover (intraday)": cross_state,
            "Quantity": None if pd.isna(qty) else float(qty),
            "Purchase Price": None if pd.isna(buy) else float(buy),
            "Stop Level": None if pd.isna(stop) else float(stop),
            "Target Price": None if pd.isna(target) else float(target),
            "P/L %": None if pl_pct is None else round(pl_pct, 2),
            "P/L Amount": None if pl_amt is None else round(pl_amt, 2),
            "Below Stop?": below_stop if below_stop is not None else "",
            "Notes": notes,
        })

df_table = pd.DataFrame(rows)

# ---------------------------------
# Alerts panel
# ---------------------------------
st.subheader("🔔 Alerts")
if alert_items:
    for msg in alert_items: st.warning(msg)
else:
    st.info("No new alerts this refresh.")

# ---------------------------------
# Display table
# ---------------------------------
if not df_table.empty:
    st.subheader("Portfolio & Intraday Signals")
    invested = df_table.dropna(subset=["Quantity","Purchase Price"])
    current_val = float((invested["Current Price"] * invested["Quantity"]).sum()) if not invested.empty else 0.0
    invested_val = float((invested["Purchase Price"] * invested["Quantity"]).sum()) if not invested.empty else 0.0
    net_pl_amt = current_val - invested_val
    net_pl_pct = (net_pl_amt / invested_val * 100.0) if invested_val else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Invested", f"{invested_val:,.2f}")
    c2.metric("Current", f"{current_val:,.2f}")
    c3.metric("Net P/L", f"{net_pl_amt:,.2f}")
    c4.metric("Net P/L %", f"{net_pl_pct:,.2f}%")

    sort_col = st.selectbox("Sort by", ["Symbol","P/L %","P/L Amount","2D %","7D %","5D %","Price vs 200 EMA","MACD Crossover (intraday)","Target Price"], index=1)
    ascending = st.checkbox("Ascending sort", value=False)
    if sort_col in df_table.columns:
        df_table = df_table.sort_values(by=sort_col, ascending=ascending, na_position="last")

    st.dataframe(df_table, use_container_width=True)

    persist_cols = ["Symbol","Quantity","Purchase Price","Stop Level","Target Price","Notes"]
    st.download_button("Download Portfolio CSV", st.session_state.portfolio[persist_cols].to_csv(index=False).encode("utf-8"), file_name="portfolio.csv")
    st.download_button("Download Full Report CSV", df_table.to_csv(index=False).encode("utf-8"), file_name="tracker_report.csv")
else:
    st.info("No rows to show yet. Add symbols or check inputs above.")
