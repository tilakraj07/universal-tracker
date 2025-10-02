# ==============================================
# UNIVERSAL TRACKER — EMA200 (3h) + MACD (daily trend) + Alerts (12h throttle)
# with 15m/1m prices for refresh
# ==============================================

import os
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
ALERTS_LOG = "alerts_log.csv"

# ---------------------------------
# Telegram setup (HARDCODED)
# ---------------------------------
tg_enable = True
tg_token  = "8298370446:AAHQJdZpq1TZumNG3tacLpBnH6Ge6cCJU3o"
tg_chat   = "888880398"

def send_telegram(msg: str):
    if not (tg_enable and tg_token and tg_chat): return
    try:
        url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
        requests.post(url, json={"chat_id": tg_chat, "text": msg})
    except Exception: pass

# ---------------------------------
# Alerts log persistence (12h throttle)
# ---------------------------------
def load_alert_log():
    if os.path.exists(ALERTS_LOG):
        try: return pd.read_csv(ALERTS_LOG, parse_dates=["timestamp"])
        except: pass
    return pd.DataFrame(columns=["key","timestamp"])

def save_alert_log(df): 
    try: df.to_csv(ALERTS_LOG, index=False)
    except: pass

alert_log = load_alert_log()
THROTTLE_HOURS = 12

def alert_recent(key):
    if alert_log.empty: return False
    recent = alert_log[alert_log["key"]==key]
    if recent.empty: return False
    return (datetime.now()-recent["timestamp"].max()) < timedelta(hours=THROTTLE_HOURS)

def record_alert(key):
    global alert_log
    alert_log = pd.concat([alert_log, pd.DataFrame([{"key":key,"timestamp":datetime.now()}])], ignore_index=True)
    save_alert_log(alert_log)

# ---------------------------------
# Data fetchers
# ---------------------------------
def ensure_cols(df):
    if isinstance(df.columns,pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

@st.cache_data(ttl=900)
def fetch_3h(sym):
    df = yf.download(sym, period="730d", interval="180m", progress=False, auto_adjust=False)
    return ensure_cols(df).dropna() if not df.empty else df

@st.cache_data(ttl=900)
def fetch_60m(sym):
    df = yf.download(sym, period="730d", interval="60m", progress=False, auto_adjust=False)
    return ensure_cols(df).dropna() if not df.empty else df

@st.cache_data(ttl=900)
def fetch_daily(sym, period="400d"):
    df = yf.download(sym, period=period, interval="1d", progress=False, auto_adjust=False)
    return ensure_cols(df).dropna() if not df.empty else df

@st.cache_data(ttl=900)
def fetch_15m_price(sym):
    df = yf.download(sym, period="7d", interval="15m", progress=False, auto_adjust=False)
    return float(df["Close"].iloc[-1]) if not df.empty else None

@st.cache_data(ttl=60)
def fetch_1m_price(sym):
    df = yf.download(sym, period="2d", interval="1m", progress=False, auto_adjust=False)
    return float(df["Close"].iloc[-1]) if not df.empty else None

def is_crypto(sym): return str(sym).upper().endswith("-USD")

# ---------------------------------
# Indicators
# ---------------------------------
def ema(s, span): return s.ewm(span=span, adjust=False).mean()

def macd(series):
    ema12, ema26 = ema(series,12), ema(series,26)
    macd_line = ema12-ema26
    signal = ema(macd_line,9)
    hist = macd_line-signal
    return macd_line, signal, hist

def macd_trend_daily(sym):
    df = fetch_daily(sym)
    if df.empty or "Close" not in df: return "None",0,0,0
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close)<30: return "None",0,0,0
    macd_line, signal, hist = macd(close)
    c,s,h = float(macd_line.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])
    return ("Bullish" if c>s else "Bearish" if c<s else "None"), c,s,h

def ema200_3h(sym):
    df3 = fetch_3h(sym)
    if not df3.empty and len(df3)>=210:
        close = pd.to_numeric(df3["Close"], errors="coerce").dropna()
        return float(ema(close,200).iloc[-1])
    # fallback: resample from 60m → 180m
    df60 = fetch_60m(sym)
    if not df60.empty:
        close = pd.to_numeric(df60["Close"], errors="coerce").dropna()
        s = pd.Series(close.values, index=pd.to_datetime(close.index))
        s3h = s.resample("180min").last().dropna()
        if len(s3h)>=210:
            return float(ema(s3h,200).iloc[-1])
    return None

# ---------------------------------
# Portfolio
# ---------------------------------
DEFAULT_SYMBOLS = ["RELIANCE.NS","HDFCBANK.NS","TCS.NS","BTC-USD","XAUUSD=X"]

def load_portfolio():
    if os.path.exists(PORTFOLIO_CSV):
        try: return pd.read_csv(PORTFOLIO_CSV)
        except: pass
    return pd.DataFrame({
        "Symbol":DEFAULT_SYMBOLS,"Quantity":np.nan,"Purchase Price":np.nan,
        "Stop Level":np.nan,"Target Price":np.nan,"Notes":""
    })

if "portfolio" not in st.session_state:
    st.session_state.portfolio = load_portfolio()

st.subheader("Edit your portfolio")
edited = st.data_editor(st.session_state.portfolio,num_rows="dynamic",use_container_width=True)
st.session_state.portfolio = edited.copy()
st.session_state.portfolio.to_csv(PORTFOLIO_CSV,index=False)

# ---------------------------------
# Main Loop
# ---------------------------------
rows, alerts = [], []

with st.spinner("Fetching data…"):
    for _,row in edited.iterrows():
        sym = str(row["Symbol"]).strip()
        if not sym: continue

        # EMA200 (3h)
        ema200_val = ema200_3h(sym)
        ema_state = "Above" if ema200_val and fetch_15m_price(sym) and fetch_15m_price(sym)>ema200_val else "Below" if ema200_val else "N/A"

        # Price from fast feed
        last_price = fetch_1m_price(sym) if is_crypto(sym) else fetch_15m_price(sym)

        # MACD daily
        macd_state, macd_val, macd_sig, macd_hist = macd_trend_daily(sym)

        # Daily pct change
        df_daily = fetch_daily(sym, period="90d")
        pct2=pct5=pct7=None
        if not df_daily.empty:
            close = pd.to_numeric(df_daily["Close"],errors="coerce").dropna()
            if len(close)>=3: pct2=(close.iloc[-1]/close.iloc[-3]-1)*100
            if len(close)>=6: pct5=(close.iloc[-1]/close.iloc[-6]-1)*100
            if len(close)>=8: pct7=(close.iloc[-1]/close.iloc[-8]-1)*100

        qty,buy,stop,target,notes = row.get("Quantity",np.nan),row.get("Purchase Price",np.nan),\
                                    row.get("Stop Level",np.nan),row.get("Target Price",np.nan),row.get("Notes","")
        pl_pct = (last_price/buy-1)*100 if last_price and pd.notna(buy) and buy!=0 else None
        pl_amt = (last_price-buy)*qty if last_price and pd.notna(buy) and pd.notna(qty) else None

        below_stop = last_price and pd.notna(stop) and stop>0 and last_price<=stop
        above_target = last_price and pd.notna(target) and target>0 and last_price>=target

        def push_alert(key,msg):
            if not alert_recent(f"{sym}:{key}"):
                alerts.append(msg); send_telegram(msg); record_alert(f"{sym}:{key}")

        if pl_pct is not None:
            if pl_pct>=10: push_alert("PL+10",f"🔔 {sym}: P/L +{pl_pct:.2f}%")
            elif pl_pct<=-10: push_alert("PL-10",f"🔔 {sym}: P/L {pl_pct:.2f}%")
            if 5<=pl_pct<10: push_alert("PL+5",f"🔔 {sym}: P/L +{pl_pct:.2f}%")
            elif -10<pl_pct<=-5: push_alert("PL-5",f"🔔 {sym}: P/L {pl_pct:.2f}%")

        if pct2 is not None:
            if pct2>=3: push_alert("2D+3",f"📈 {sym}: +{pct2:.2f}% in 2 days")
            elif pct2<=-3: push_alert("2D-3",f"📉 {sym}: {pct2:.2f}% in 2 days")

        if below_stop: push_alert("STOP",f"⛔ {sym}: {last_price:.2f} ≤ Stop {stop:.2f}")
        if above_target: push_alert("TARGET",f"🎯 {sym}: {last_price:.2f} ≥ Target {target:.2f}")

        rows.append({
            "Symbol":sym,"Date/Time":datetime.now(),"Current Price":None if not last_price else round(last_price,4),
            "200 EMA (3h)":None if ema200_val is None else round(ema200_val,4),
            "Price vs 200 EMA (3h)":ema_state,
            "MACD Trend (daily)":macd_state,
            "2D %":None if pct2 is None else round(pct2,2),
            "5D %":None if pct5 is None else round(pct5,2),
            "7D %":None if pct7 is None else round(pct7,2),
            "Purchase Price":None if pd.isna(buy) else float(buy),
            "Stop Level":None if pd.isna(stop) else float(stop),
            "Target Price":None if pd.isna(target) else float(target),
            "P/L %":None if pl_pct is None else round(pl_pct,2),
            "P/L Amount":None if pl_amt is None else round(pl_amt,2),
            "Notes":notes
        })

df_table = pd.DataFrame(rows)

# Alerts panel
st.subheader("🔔 Alerts (12h throttle)")
c1,c2=st.columns([3,1])
with c1:
    if alerts: [st.warning(m) for m in alerts]
    else: st.info("No new alerts this refresh.")
with c2:
    if st.button("🔄 Reset Alerts"):
        if os.path.exists(ALERTS_LOG): os.remove(ALERTS_LOG)
        alert_log=pd.DataFrame(columns=["key","timestamp"])
        st.success("Alert history cleared ✅")

# Display
if not df_table.empty:
    st.dataframe(df_table,use_container_width=True)
    st.download_button("Download CSV",df_table.to_csv(index=False).encode("utf-8"),file_name="tracker.csv")
else:
    st.info("No rows to show yet.")
