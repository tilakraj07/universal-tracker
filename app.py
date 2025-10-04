# ==============================================
# UNIVERSAL TRACKER — EMA200 (3h) + MACD (daily) + Alerts (12h throttle)
# Durable storage via Google Sheets (CSV fallback) + Save/Reload buttons
# NOW with cached Google Sheets reads to avoid 429 rate limits
# ==============================================

import os
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import requests
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# Google Sheets libs
import gspread
from gspread_dataframe import set_with_dataframe, get_as_dataframe
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="Universal Tracker — Intraday + Alerts", layout="wide")
st.title("📊 Universal Tracker — EMA200 (3h) + MACD (daily trend) + Alerts")

# Auto-refresh (15 min). If you still hit 429, consider 30 min: 1_800_000
st_autorefresh(interval=900_000, key="auto15min")

PORTFOLIO_CSV = "portfolio.csv"
ALERTS_LOG = "alerts_log.csv"

# ---------------------------------
# Telegram (hardcoded; optional)
# ---------------------------------
tg_enable = True
tg_token  = "8298370446:AAHQJdZpq1TZumNG3tacLpBnH6Ge6cCJU3o"
tg_chat   = "888880398"

def send_telegram(msg: str):
    if not (tg_enable and tg_token and tg_chat):
        return
    try:
        url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
        requests.post(url, json={"chat_id": tg_chat, "text": msg}, timeout=15)
    except Exception:
        pass

# ---------------------------------
# Rerun helper (works on all Streamlit versions)
# ---------------------------------
def _safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()

# ---------------------------------
# Google Sheets helpers (Sheets first, CSV fallback)
# ---------------------------------
_SHEETS_CLIENT = None

def _get_sheets_client():
    """Lazy-init Google Sheets client from st.secrets."""
    global _SHEETS_CLIENT
    if _SHEETS_CLIENT is not None:
        return _SHEETS_CLIENT
    try:
        sa_info = st.secrets["gcp_service_account"]
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            # If open_by_url fails, uncomment Drive scope below and enable Drive API in GCP:
            # "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(sa_info, scopes=scopes)
        gc = gspread.authorize(creds)
        _SHEETS_CLIENT = gc
        return gc
    except Exception:
        return None

def _open_ws(url: str):
    gc = _get_sheets_client()
    if not gc or not url:
        return None
    try:
        sh = gc.open_by_url(url)
        return sh.sheet1
    except Exception:
        return None

def sheets_available():
    try:
        return (
            "sheets" in st.secrets
            and "portfolio_sheet_url" in st.secrets["sheets"]
            and "alerts_sheet_url" in st.secrets["sheets"]
            and _get_sheets_client() is not None
        )
    except Exception:
        return False

# ---------------------------------
# Cached reads from Google Sheets (to avoid 429 rate limits)
# ---------------------------------
CACHE_TTL_SHEETS = 300  # 5 minutes

@st.cache_data(ttl=CACHE_TTL_SHEETS, show_spinner=False)
def _read_portfolio_from_sheets(url: str):
    ws = _open_ws(url)
    if not ws:
        return pd.DataFrame()
    df = get_as_dataframe(ws, evaluate_formulas=True, header=0)
    return df.dropna(how="all")

@st.cache_data(ttl=CACHE_TTL_SHEETS, show_spinner=False)
def _read_alerts_from_sheets(url: str):
    ws = _open_ws(url)
    if not ws:
        return pd.DataFrame(columns=["key", "timestamp"])
    df = get_as_dataframe(ws, evaluate_formulas=True, header=0)
    return df.dropna(how="all")

# ---------------------------------
# Portfolio persistence (Sheets + CSV fallback)
# ---------------------------------
DEFAULT_SYMBOLS = [
    "RELIANCE.NS","HDFCBANK.NS","TCS.NS",  # India
    "BTC-USD","SOL-USD",                   # Crypto
    "XAUUSD=X","XAGUSD=X"                  # Spot metals
]

def _ensure_portfolio_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Symbol","Quantity","Purchase Price","Stop Level","Target Price","Notes"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan if c != "Notes" else ""
    return df[cols]

def load_portfolio():
    if sheets_available():
        try:
            df = _read_portfolio_from_sheets(st.secrets["sheets"]["portfolio_sheet_url"])
            if not df.empty:
                return _ensure_portfolio_cols(df)
        except Exception:
            pass
    if os.path.exists(PORTFOLIO_CSV):
        try:
            df = pd.read_csv(PORTFOLIO_CSV)
            return _ensure_portfolio_cols(df)
        except Exception:
            pass
    return pd.DataFrame({
        "Symbol": DEFAULT_SYMBOLS,
        "Quantity": np.nan,
        "Purchase Price": np.nan,
        "Stop Level": np.nan,
        "Target Price": np.nan,
        "Notes": ""
    })

def save_portfolio(df: pd.DataFrame):
    # Always save CSV as local fallback
    try:
        df.to_csv(PORTFOLIO_CSV, index=False)
    except Exception:
        pass
    # Write to Sheets
    if sheets_available():
        ws = _open_ws(st.secrets["sheets"]["portfolio_sheet_url"])
        if ws:
            try:
                ws.clear()
                set_with_dataframe(ws, df)
                # Bust the read cache so next read gets fresh data
                _read_portfolio_from_sheets.clear()
            except Exception:
                pass

# ---------------------------------
# Alerts log persistence (Sheets + CSV fallback)
# ---------------------------------
def load_alert_log():
    if sheets_available():
        try:
            df = _read_alerts_from_sheets(st.secrets["sheets"]["alerts_sheet_url"])
            if not df.empty:
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                else:
                    df["timestamp"] = pd.NaT
                if "key" not in df.columns:
                    df["key"] = ""
                return df[["key","timestamp"]]
        except Exception:
            pass
    if os.path.exists(ALERTS_LOG):
        try:
            return pd.read_csv(ALERTS_LOG, parse_dates=["timestamp"])
        except Exception:
            pass
    return pd.DataFrame(columns=["key","timestamp"])

def save_alert_log(df: pd.DataFrame):
    try:
        df.to_csv(ALERTS_LOG, index=False)
    except Exception:
        pass
    if sheets_available():
        ws = _open_ws(st.secrets["sheets"]["alerts_sheet_url"])
        if ws:
            try:
                ws.clear()
                df2 = df.copy()
                if "timestamp" in df2.columns:
                    df2["timestamp"] = df2["timestamp"].astype(str)
                set_with_dataframe(ws, df2)
                _read_alerts_from_sheets.clear()
            except Exception:
                pass

# Initialize alerts log once
alert_log = load_alert_log()
THROTTLE_HOURS = 12

def alert_recent(key):
    if alert_log.empty:
        return False
    recent = alert_log[alert_log["key"] == key]
    if recent.empty:
        return False
    return (datetime.now() - recent["timestamp"].max()) < timedelta(hours=THROTTLE_HOURS)

def record_alert(key):
    global alert_log
    alert_log = pd.concat(
        [alert_log, pd.DataFrame([{"key": key, "timestamp": datetime.now()}])],
        ignore_index=True
    )
    save_alert_log(alert_log)

# ---------------------------------
# Data fetchers
# ---------------------------------
def ensure_cols(df):
    if isinstance(df.columns, pd.MultiIndex):
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

def is_crypto(sym):
    return str(sym).upper().endswith("-USD")

# ---------------------------------
# Indicators
# ---------------------------------
def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def macd(series):
    ema12, ema26 = ema(series,12), ema(series,26)
    macd_line = ema12 - ema26
    signal = ema(macd_line,9)
    hist = macd_line - signal
    return macd_line, signal, hist

def macd_trend_daily(sym):
    df = fetch_daily(sym)
    if df.empty or "Close" not in df:
        return "None", 0, 0, 0
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close) < 30:
        return "None", 0, 0, 0
    macd_line, signal, hist = macd(close)
    c, s, h = float(macd_line.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])
    return ("Bullish" if c > s else "Bearish" if c < s else "None"), c, s, h

def ema200_3h(sym):
    df3 = fetch_3h(sym)
    if not df3.empty and len(df3) >= 210:
        close = pd.to_numeric(df3["Close"], errors="coerce").dropna()
        return float(ema(close, 200).iloc[-1])
    # fallback: resample from 60m → 180m
    df60 = fetch_60m(sym)
    if not df60.empty:
        close = pd.to_numeric(df60["Close"], errors="coerce").dropna()
        s = pd.Series(close.values, index=pd.to_datetime(close.index))
        s3h = s.resample("180min").last().dropna()
        if len(s3h) >= 210:
            return float(ema(s3h, 200).iloc[-1])
    return None

# ---------------------------------
# Portfolio UI (Save/Reload; no auto-overwrite)
# ---------------------------------
def _clean_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    if "Symbol" not in clean.columns:
        clean["Symbol"] = ""
    clean["Symbol"] = clean["Symbol"].astype(str).str.strip()
    clean = clean[clean["Symbol"] != ""]
    clean = clean.drop_duplicates(subset=["Symbol"], keep="last").reset_index(drop=True)
    return _ensure_portfolio_cols(clean)

if "portfolio" not in st.session_state:
    st.session_state.portfolio = load_portfolio()

st.subheader("Edit your portfolio")
edited = st.data_editor(
    st.session_state.portfolio,
    num_rows="dynamic",
    use_container_width=True,
    key="portfolio_editor",
)

c_save, c_reload = st.columns([1,1])
with c_save:
    if st.button("💾 Save portfolio", use_container_width=True):
        clean = _clean_portfolio(edited)
        st.session_state.portfolio = clean
        save_portfolio(clean)
        st.success("Portfolio saved ✅")
        _safe_rerun()

with c_reload:
    if st.button("🔁 Reload from storage", use_container_width=True):
        # Clear only the Sheets read caches; next load will refetch once
        _read_portfolio_from_sheets.clear()
        _read_alerts_from_sheets.clear()
        st.session_state.portfolio = load_portfolio()
        _safe_rerun()

# ---------------------------------
# Main data build
# ---------------------------------
rows, alerts = [], []

with st.spinner("Fetching data…"):
    for _, row in st.session_state.portfolio.iterrows():
        sym = str(row["Symbol"]).strip()
        if not sym:
            continue

        ema200_val = ema200_3h(sym)
        last_price = fetch_1m_price(sym) if is_crypto(sym) else fetch_15m_price(sym)

        ema_state = "N/A"
        if (ema200_val is not None) and (last_price is not None):
            ema_state = "Above" if last_price > ema200_val else "Below"

        # compute MACD trend ONLY (do not expose numeric MACD columns in UI)
        macd_state, macd_val, macd_sig, macd_hist = macd_trend_daily(sym)

        df_daily = fetch_daily(sym, period="90d")
        pct2 = pct5 = pct7 = None
        if not df_daily.empty:
            close = pd.to_numeric(df_daily["Close"], errors="coerce").dropna()
            if len(close) >= 3: pct2 = (close.iloc[-1] / close.iloc[-3] - 1) * 100
            if len(close) >= 6: pct5 = (close.iloc[-1] / close.iloc[-6] - 1) * 100
            if len(close) >= 8: pct7 = (close.iloc[-1] / close.iloc[-8] - 1) * 100

        qty    = row.get("Quantity", np.nan)
        buy    = row.get("Purchase Price", np.nan)
        stop   = row.get("Stop Level", np.nan)
        target = row.get("Target Price", np.nan)
        notes  = row.get("Notes", "")

        pl_pct = (last_price / buy - 1) * 100 if (last_price is not None and pd.notna(buy) and buy != 0) else None
        pl_amt = (last_price - buy) * qty if (last_price is not None and pd.notna(buy) and pd.notna(qty)) else None
        invested_amt = (qty * buy) if (pd.notna(qty) and pd.notna(buy)) else None  # NEW: Invested Amount

        below_stop   = (last_price is not None) and pd.notna(stop) and stop > 0 and last_price <= stop
        above_target = (last_price is not None) and pd.notna(target) and target > 0 and last_price >= target

        def push_alert(key, msg):
            if not alert_recent(f"{sym}:{key}"):
                alerts.append(msg)
                send_telegram(msg)
                record_alert(f"{sym}:{key}")

        if pl_pct is not None:
            if pl_pct >= 10: push_alert("PL+10", f"🔔 {sym}: P/L +{pl_pct:.2f}%")
            elif pl_pct <= -10: push_alert("PL-10", f"🔔 {sym}: P/L {pl_pct:.2f}%")
            if 5 <= pl_pct < 10: push_alert("PL+5", f"🔔 {sym}: P/L +{pl_pct:.2f}%")
            elif -10 < pl_pct <= -5: push_alert("PL-5", f"🔔 {sym}: P/L {pl_pct:.2f}%")

        if pct2 is not None:
            if pct2 >= 3: push_alert("2D+3", f"📈 {sym}: +{pct2:.2f}% in 2 days")
            elif pct2 <= -3: push_alert("2D-3", f"📉 {sym}: {pct2:.2f}% in 2 days")

        if below_stop:   push_alert("STOP",   f"⛔ {sym}: {last_price:.2f} ≤ Stop {float(stop):.2f}")
        if above_target: push_alert("TARGET", f"🎯 {sym}: {last_price:.2f} ≥ Target {float(target):.2f}")

        signal_note = ""
        if pd.notna(pl_pct):
            if pl_pct <= -10: signal_note = "Loss >10%"
            elif pl_pct <= -5: signal_note = "Loss >5%"
            elif pl_pct >= 10: signal_note = "Gain >10%"
            elif pl_pct >= 5:  signal_note = "Gain >5%"

        if signal_note == "" and pd.notna(pct2) and (abs(pct2) >= 3):
            signal_note = "2D move >3%"

        # NOTE: Removed MACD numeric columns from UI; kept MACD Trend.
        rows.append({
            "Symbol": sym,
            "Signal": signal_note,
            "P/L %": None if pl_pct is None else round(pl_pct, 2),  # moved next to Signal in UI (reordered later)
            "2D %": None if pct2 is None else round(pct2, 2),
            "5D %": None if pct5 is None else round(pct5, 2),
            "7D %": None if pct7 is None else round(pct7, 2),
            "Current Price": None if last_price is None else round(last_price, 2),
            "200 EMA (3h)": None if ema200_val is None else round(ema200_val, 2),
            "Price vs 200 EMA (3h)": ema_state,
            "MACD Trend (daily)": macd_state,
            "Invested Amount": None if invested_amt is None else round(float(invested_amt), 2),  # NEW position column
            "Quantity": None if pd.isna(qty) else round(float(qty), 2),
            "Purchase Price": None if pd.isna(buy) else round(float(buy), 2),
            "Stop Level": None if pd.isna(stop) else round(float(stop), 2),
            "Target Price": None if pd.isna(target) else round(float(target), 2),
            "P/L Amount": None if pl_amt is None else round(pl_amt, 2),
            "Notes": notes
        })

df_table = pd.DataFrame(rows)

# ======= Header Summary (Overall — only count rows with Purchase Price) =======
def _num(x):
    try: return float(x)
    except: return np.nan

if not df_table.empty:
    # Work on a copy, coerce numerics
    df_sum = df_table.copy()
    for c in ["Quantity","Purchase Price","Current Price","Invested Amount"]:
        if c in df_sum.columns:
            df_sum[c] = df_sum[c].apply(_num)

    # Keep only positions where a Purchase Price is provided and Quantity > 0
    mask_positions = df_sum["Purchase Price"].notna() & df_sum["Quantity"].notna() & (df_sum["Quantity"] > 0)
    df_pos = df_sum.loc[mask_positions].copy()

    # Compute invested/current values for these positions only
    if "Invested Amount" not in df_pos.columns:
        df_pos["Invested Amount"] = df_pos["Quantity"] * df_pos["Purchase Price"]
    df_pos["Current Value"] = df_pos["Quantity"] * df_pos["Current Price"]

    # Totals (only purchased positions)
    total_invested = float(np.nansum(df_pos["Invested Amount"])) if not df_pos.empty else 0.0
    total_current  = float(np.nansum(df_pos["Current Value"])) if not df_pos.empty else 0.0
    overall_ret_pct = (total_current/total_invested - 1) * 100 if total_invested > 0 else 0.0

    # Daily portfolio value change % (position-weighted, only purchased positions)
    daily_prev_total = 0.0
    daily_curr_total = 0.0
    for _, r in df_pos.iterrows():
        q = _num(r.get("Quantity"))
        sym = str(r.get("Symbol", "")).strip()
        if pd.isna(q) or q is None or q == 0 or not sym:
            continue
        dfd = fetch_daily(sym, period="10d")
        if not dfd.empty and "Close" in dfd:
            closes = pd.to_numeric(dfd["Close"], errors="coerce").dropna()
            if len(closes) >= 2:
                prev = float(closes.iloc[-2])
                curr = float(closes.iloc[-1])
                daily_prev_total += q * prev
                daily_curr_total += q * curr
            elif len(closes) == 1:
                curr = float(closes.iloc[-1])
                daily_prev_total += q * curr
                daily_curr_total += q * curr
    daily_change_pct = (daily_curr_total / daily_prev_total - 1) * 100 if daily_prev_total > 0 else 0.0

    # Top 3 holdings (%) among purchased positions
    top3_share_pct = 0.0
    if total_current > 0 and not df_pos.empty:
        weights = (df_pos.assign(weight=lambda d: d["Current Value"]/total_current * 100)
                        .sort_values("weight", ascending=False)["weight"])
        top3_share_pct = float(weights.head(3).sum())

    # Render header metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Overall Portfolio Value", f"{total_current:,.2f}")
    m2.metric("Overall % (Invested vs Current)", f"{overall_ret_pct:+.2f}%")
    m3.metric("Daily Portfolio Change", f"{daily_change_pct:+.2f}%")
    m4.metric("Top 3 Holdings (share)", f"{top3_share_pct:.2f}%")

# Ensure Signal & P/L % placement:
# - Signal immediately after Symbol (already handled)
# - P/L % right after Signal
if not df_table.empty and "Signal" in df_table.columns and "Symbol" in df_table.columns and "P/L %" in df_table.columns:
    cols = df_table.columns.tolist()
    # move Signal next to Symbol (keep existing behavior)
    cols.insert(cols.index("Symbol") + 1, cols.pop(cols.index("Signal")))
    # move P/L % right after Signal
    cols.insert(cols.index("Symbol") + 2, cols.pop(cols.index("P/L %")))
    df_table = df_table[cols]

# Highlight Symbol column
def highlight_symbol(row):
    red = pd.notna(row.get("P/L %")) and row["P/L %"] <= -5
    green = pd.notna(row.get("P/L %")) and row["P/L %"] >= 5
    blue = pd.notna(row.get("2D %")) and abs(row["2D %"]) >= 3
    color = ""
    if red: color = "background-color: lightcoral"
    elif green: color = "background-color: lightgreen"
    elif blue: color = "background-color: lightblue"
    return [color if col == "Symbol" else "" for col in df_table.columns]

format_dict = {"2D %": "{:.2f}%", "5D %": "{:.2f}%", "7D %": "{:.2f}%", "P/L %": "{:.2f}%"}
styled_table = df_table.style.apply(highlight_symbol, axis=1).format(format_dict).format(precision=2)

# ---------------------------------
# Alerts Panel with Reset (clears Sheets + CSV)
# ---------------------------------
st.subheader("🔔 Alerts (12h throttle)")
c1, c2 = st.columns([3, 1])
with c1:
    if alerts:
        for msg in alerts:
            st.warning(msg)
    else:
        st.info("No new alerts this refresh.")
with c2:
    if st.button("🔄 Reset Alerts"):
        empty_df = pd.DataFrame(columns=["key","timestamp"])
        try:
            if os.path.exists(ALERTS_LOG):
                os.remove(ALERTS_LOG)
        except Exception:
            pass
        save_alert_log(empty_df)
        # reset in-memory
        alert_log = empty_df
        st.success("Alert history cleared ✅")

# ---------------------------------
# Portfolio Display
# ---------------------------------
if not df_table.empty:
    st.subheader("Portfolio & Signals")
    st.dataframe(styled_table, use_container_width=True)
    st.download_button(
        "Download Full Report CSV",
        df_table.to_csv(index=False).encode("utf-8"),
        file_name="tracker_report.csv"
    )
else:
    st.info("No rows to show yet.")

# ---------------------------------
# Google Sheets connection check (only runs on click to save quota)
# ---------------------------------
with st.expander("🔎 Google Sheets connection check", expanded=False):
    if st.button("Run check"):
        try:
            sa = st.secrets["gcp_service_account"]
            st.write("✓ Secrets loaded")
            st.write("Service account:", sa.get("client_email", "N/A"))
        except Exception as e:
            st.error(f"Secrets not loaded: {e}")

        try:
            creds = Credentials.from_service_account_info(
                st.secrets["gcp_service_account"],
                scopes=["https://www.googleapis.com/auth/spreadsheets"]
            )
            gc = gspread.authorize(creds)
            sh = gc.open_by_url(st.secrets["sheets"]["portfolio_sheet_url"])
            st.success(f"✓ Can open Portfolio sheet: {sh.title}")
            sh2 = gc.open_by_url(st.secrets["sheets"]["alerts_sheet_url"])
            st.success(f"✓ Can open Alerts sheet: {sh2.title}")
        except Exception as e:
            st.error(f"Sheets error: {e}")
    else:
        st.caption("Click ‘Run check’ to test access (avoids consuming API quota on each refresh).")
