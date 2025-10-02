# ==============================================
# UNIVERSAL TRACKER — EMA200 (3h) + MACD (daily trend) + Alerts (12h throttle)
# ==============================================
# HOW TO RUN:
#   pip install streamlit streamlit-autorefresh yfinance pandas numpy requests
#   python -m streamlit run app.py
# ==============================================

import os
import hashlib
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import requests
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Universal Tracker — Intraday + Alerts", layout="wide")
st.title("📊 Universal Tracker — EMA200 (3h) + MACD (daily trend) + Alerts")

# Auto-refresh every 15 minutes
st_autorefresh(interval=900_000, key="auto15min")

PORTFOLIO_CSV = "portfolio.csv"
ALERTS_LOG = "alerts_log.csv"   # persists alert history

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
        pass

# ---------------------------------
# Alerts log persistence (12h throttle)
# ---------------------------------
def load_alert_log() -> pd.DataFrame:
    if os.path.exists(ALERTS_LOG):
        try:
            df = pd.read_csv(ALERTS_LOG, parse_dates=["timestamp"])
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=["key", "timestamp"])

def save_alert_log(df: pd.DataFrame):
    try:
        df.to_csv(ALERTS_LOG, index=False)
    except Exception:
        pass

alert_log = load_alert_log()
THROTTLE_HOURS = 12  # limit per rule per asset

def alert_recent(key: str) -> bool:
    if alert_log.empty:
        return False
    recent = alert_log[alert_log["key"] == key]
    if recent.empty:
        return False
    last_time = recent["timestamp"].max()
    return (datetime.now() - last_time) < timedelta(hours=THROTTLE_HOURS)

def record_alert(key: str):
    global alert_log
    new_row = pd.DataFrame([{"key": key, "timestamp": datetime.now()}])
    alert_log = pd.concat([alert_log, new_row], ignore_index=True)
    save_alert_log(alert_log)

# ---------------------------------
# Data fetchers
# ---------------------------------
def asset_kind(sym: str) -> str:
    s = sym.upper()
    if s.endswith(".NS") or s.endswith(".BO"): return "stock"
    if s.endswith("-USD"): return "crypto"
    if s.endswith("=F") or s in {"GC=F","SI=F"}: return "futures"
    if s.endswith("=X"): return "fx"
    return "other"

@st.cache_data(ttl=900, show_spinner=False)
def fetch_3h(ticker: str, period="400d", interval="180m") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=900, show_spinner=False)
def fetch_daily_bars(ticker: str, period="400d", interval="1d") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df.dropna()
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=900, show_spinner=False)
def fetch_daily_pct(ticker: str, period="90d", interval="1d") -> pd.DataFrame:
    return fetch_daily_bars(ticker, period=period, interval=interval)

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def macd(series: pd.Series):
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist

# ---------------------------------
# MACD Trend State (daily)
# ---------------------------------
def macd_trend_daily(sym: str):
    df = fetch_daily_bars(sym, period="400d", interval="1d")
    if df.empty or "Close" not in df.columns or len(df) < 30:
        return "None", 0.0, 0.0, 0.0
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close) < 30:
        return "None", 0.0, 0.0, 0.0
    macd_line, signal, hist = macd(close)
    curr_macd = float(macd_line.iloc[-1])
    curr_sig  = float(signal.iloc[-1])
    hist_val  = float(hist.iloc[-1])
    if curr_macd > curr_sig:
        state = "Bullish"
    elif curr_macd < curr_sig:
        state = "Bearish"
    else:
        state = "None"
    return state, curr_macd, curr_sig, hist_val

# ---------------------------------
# Defaults & portfolio
# ---------------------------------
DEFAULT_SYMBOLS = ["RELIANCE.NS","HDFCBANK.NS","TCS.NS","BTC-USD","XAUUSD=X"]

def load_or_init_portfolio() -> pd.DataFrame:
    if os.path.exists(PORTFOLIO_CSV):
        try:
            df = pd.read_csv(PORTFOLIO_CSV)
            for col in ["Symbol","Quantity","Purchase Price","Stop Level","Target Price","Notes"]:
                if col not in df.columns: df[col] = np.nan if col!="Notes" else ""
            return df
        except: pass
    return pd.DataFrame({
        "Symbol": DEFAULT_SYMBOLS,
        "Quantity": np.nan,
        "Purchase Price": np.nan,
        "Stop Level": np.nan,
        "Target Price": np.nan,
        "Notes": ""
    })

if "portfolio" not in st.session_state:
    st.session_state.portfolio = load_or_init_portfolio()

# ---------------------------------
# Portfolio editor
# ---------------------------------
st.subheader("Edit your portfolio")
edited = st.data_editor(st.session_state.portfolio, num_rows="dynamic", use_container_width=True)
st.session_state.portfolio = edited.copy()
st.session_state.portfolio.to_csv(PORTFOLIO_CSV, index=False)

# ---------------------------------
# Main Loop
# ---------------------------------
rows, failed, alert_items = [], [], []

with st.spinner("Fetching data…"):
    for _, row in edited.iterrows():
        sym = str(row["Symbol"])
        if not sym: continue

        # --- EMA200 (3h candles) ---
        ema200_val, ema_state = None, "N/A"
        df_3h = fetch_3h(sym)
        if not df_3h.empty and "Close" in df_3h.columns and len(df_3h) >= 210:
            close_3h = pd.to_numeric(df_3h["Close"], errors="coerce").dropna()
            ema200_val = ema(close_3h, 200).iloc[-1]
            last_price = float(close_3h.iloc[-1])
            ema_state = "Above" if last_price > ema200_val else "Below"
        else:
            last_price = np.nan

        # --- MACD (daily, trend state) ---
        macd_state, macd_val, macd_sig, macd_hist = macd_trend_daily(sym)

        # --- Daily % changes (2D/5D/7D) ---
        df_daily = fetch_daily_pct(sym)
        pct2 = pct5 = pct7 = None
        if not df_daily.empty and "Close" in df_daily.columns:
            dclose = pd.to_numeric(df_daily["Close"], errors="coerce").dropna()
            if len(dclose) >= 3:
                pct2 = (dclose.iloc[-1] / dclose.iloc[-3] - 1.0) * 100
            if len(dclose) >= 6:
                pct5 = (dclose.iloc[-1] / dclose.iloc[-6] - 1.0) * 100
            if len(dclose) >= 8:
                pct7 = (dclose.iloc[-1] / dclose.iloc[-8] - 1.0) * 100

        # --- User inputs ---
        qty    = row.get("Quantity", np.nan)
        buy    = row.get("Purchase Price", np.nan)
        stop   = row.get("Stop Level", np.nan)
        target = row.get("Target Price", np.nan)
        notes  = row.get("Notes", "")

        pl_pct = (last_price/buy - 1)*100 if pd.notna(buy) and buy!=0 else None
        pl_amt = (last_price - buy)*qty if pd.notna(buy) and pd.notna(qty) else None

        below_stop = (last_price <= stop) if pd.notna(stop) and stop>0 else None
        above_target = (last_price >= target) if pd.notna(target) and target>0 else None

        # ---- Alerts ----
        def push_alert(rule_key: str, message: str):
            key = f"{sym}:{rule_key}"
            if not alert_recent(key):
                alert_items.append(message)
                send_telegram(message)
                record_alert(key)

        if pl_pct is not None:
            if pl_pct >= 10: push_alert("PL+10", f"🔔 {sym}: P/L +{pl_pct:.2f}% (≥10%)")
            elif pl_pct <= -10: push_alert("PL-10", f"🔔 {sym}: P/L {pl_pct:.2f}% (≤-10%)")
            if 5 <= pl_pct < 10: push_alert("PL+5", f"🔔 {sym}: P/L +{pl_pct:.2f}% (≥5%)")
            elif -10 < pl_pct <= -5: push_alert("PL-5", f"🔔 {sym}: P/L {pl_pct:.2f}% (≤-5%)")

        if pct2 is not None:
            if pct2 >= 3: push_alert("2D+3", f"📈 {sym}: +{pct2:.2f}% in 2 days (≥3%)")
            elif pct2 <= -3: push_alert("2D-3", f"📉 {sym}: {pct2:.2f}% in 2 days (≤-3%)")

        if below_stop is True: push_alert("STOP", f"⛔ {sym}: {last_price:.2f} ≤ Stop {stop:.2f}")
        if above_target is True: push_alert("TARGET", f"🎯 {sym}: {last_price:.2f} ≥ Target {target:.2f}")

        rows.append({
            "Symbol": sym,
            "Date/Time": datetime.now(),
            "Current Price": round(last_price,4) if pd.notna(last_price) else None,
            "2D %": None if pct2 is None else round(pct2,2),
            "5D %": None if pct5 is None else round(pct5,2),
            "7D %": None if pct7 is None else round(pct7,2),
            "200 EMA (3h)": None if ema200_val is None else round(float(ema200_val),4),
            "Price vs 200 EMA (3h)": ema_state,
            "MACD (daily)": round(macd_val,5),
            "MACD Signal (daily)": round(macd_sig,5),
            "MACD Hist (daily)": round(macd_hist,5),
            "MACD Trend (daily)": macd_state,
            "Quantity": None if pd.isna(qty) else float(qty),
            "Purchase Price": None if pd.isna(buy) else float(buy),
            "Stop Level": None if pd.isna(stop) else float(stop),
            "Target Price": None if pd.isna(target) else float(target),
            "P/L %": None if pl_pct is None else round(pl_pct,2),
            "P/L Amount": None if pl_amt is None else round(pl_amt,2),
            "Below Stop?": below_stop if below_stop is not None else "",
            "Notes": notes,
        })

df_table = pd.DataFrame(rows)

# ---------------------------------
# Alerts Panel with Reset
# ---------------------------------
st.subheader("🔔 Alerts (12h throttle)")
col1, col2 = st.columns([3,1])
with col1:
    if alert_items:
        for msg in alert_items:
            st.warning(msg)
    else:
        st.info("No new alerts this refresh (per rule per asset limited to once every 12 hours).")
with col2:
    if st.button("🔄 Reset Alerts"):
        try:
            if os.path.exists(ALERTS_LOG): os.remove(ALERTS_LOG)
            alert_log = pd.DataFrame(columns=["key","timestamp"])
            st.success("Alert history cleared ✅")
        except Exception as e:
            st.error(f"Could not clear log: {e}")

# ---------------------------------
# Portfolio Display
# ---------------------------------
if not df_table.empty:
    st.subheader("Portfolio & Signals")
    st.dataframe(df_table, use_container_width=True)
    st.download_button("Download Full Report CSV", df_table.to_csv(index=False).encode("utf-8"), file_name="tracker_report.csv")
else:
    st.info("No rows to show yet. Add symbols or check inputs above.")
