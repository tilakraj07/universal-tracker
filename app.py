# app.py
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

st.set_page_config(page_title="Universal Tracker (Free Data)", layout="wide")
st.title("Universal Tracker — Indian Stocks, Crypto, Metals (Free)")

# -----------------------------
# Robust Yahoo fetch (cached)
# -----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_history_yf(ticker: str, period="400d", interval="1d") -> pd.DataFrame:
    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].dropna()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()

# -----------------------------
# Indicators
# -----------------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def macd(series: pd.Series):
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist

def compute_snapshot(df: pd.DataFrame) -> dict:
    out = {"ok": False}
    if df.empty or len(df) < 210:  # need enough data for EMA200
        return out

    close = df["Close"]
    ema200 = ema(close, 200)
    pct5 = (close / close.shift(5) - 1.0) * 100.0

    macd_line, signal, hist = macd(close)

    bull_cross = (macd_line.iloc[-1] > signal.iloc[-1]) and (macd_line.iloc[-2] <= signal.iloc[-2])
    bear_cross = (macd_line.iloc[-1] < signal.iloc[-1]) and (macd_line.iloc[-2] >= signal.iloc[-2])
    cross_state = "Bullish" if bull_cross else ("Bearish" if bear_cross else "None")

    out.update({
        "ok": True,
        "date": close.index[-1].date(),
        "close": float(close.iloc[-1]),
        "pct5": None if np.isnan(pct5.iloc[-1]) else float(pct5.iloc[-1]),
        "ema200": float(ema200.iloc[-1]),
        "price_vs_ema200": "Above" if close.iloc[-1] > ema200.iloc[-1] else "Below",
        "macd": float(macd_line.iloc[-1]),
        "macd_signal": float(signal.iloc[-1]),
        "macd_hist": float(hist.iloc[-1]),
        "macd_cross": cross_state,
    })
    return out

# -----------------------------
# Defaults (NSE + Crypto + Gold)
# -----------------------------
DEFAULT_SYMBOLS = [
    # Indian stocks (Yahoo: .NS for NSE, .BO for BSE)
    "RELIANCE.NS", "HDFCBANK.NS", "TCS.NS", "INFY.NS", "ICICIBANK.NS",
    # Crypto
    "BTC-USD", "ETH-USD", "SOL-USD",
    # Metals (spot & futures)
    "XAUUSD=X",  # Gold spot (USD)
    "XAGUSD=X",  # Silver spot (USD)
    "GC=F",      # COMEX Gold futures
    "SI=F",      # COMEX Silver futures
]

with st.expander("How symbols work (click to open)"):
    st.markdown("""
- **Indian stocks (NSE/BSE)**: append `.NS` (NSE) or `.BO` (BSE). e.g., `RELIANCE.NS`, `SBIN.NS`, `HDFCBANK.NS`.
- **Crypto**: `BTC-USD`, `ETH-USD`, `SOL-USD`, etc.
- **Metals**: `XAUUSD=X` (gold spot), `XAGUSD=X` (silver spot); futures like `GC=F`, `SI=F`.
    """)

symbols_text = st.text_area(
    "Enter symbols (one per line)",
    value="\n".join(DEFAULT_SYMBOLS),
    height=200
).strip()
symbols = [s.strip() for s in symbols_text.splitlines() if s.strip()]

st.caption("Data via yfinance (Yahoo Finance). No API keys needed. Free usage. Timeframe: Daily.")

# -----------------------------
# Build table
# -----------------------------
rows = []
failed = []

with st.spinner("Fetching & computing…"):
    for sym in symbols:
        df = fetch_history_yf(sym, period="400d", interval="1d")
        snap = compute_snapshot(df)
        if not snap.get("ok", False):
            failed.append(sym)
            continue
        rows.append({
            "Symbol": sym,
            "Date": snap["date"],
            "Current Price": round(snap["close"], 4),
            "5D %": None if snap["pct5"] is None else round(snap["pct5"], 2),
            "200 EMA": round(snap["ema200"], 4),
            "Price vs 200 EMA": snap["price_vs_ema200"],
            "MACD": round(snap["macd"], 5),
            "MACD Signal": round(snap["macd_signal"], 5),
            "MACD Hist": round(snap["macd_hist"], 5),
            "MACD Crossover Today": snap["macd_cross"],
        })

df_table = pd.DataFrame(rows)
if not df_table.empty:
    sort_col = st.selectbox(
        "Sort by",
        ["Symbol", "5D %", "Price vs 200 EMA", "MACD Crossover Today"],
        index=1
    )
    ascending = st.checkbox("Ascending sort", value=False)
    if sort_col in df_table.columns:
        df_table = df_table.sort_values(by=sort_col, ascending=ascending, na_position="last")
    st.dataframe(df_table, use_container_width=True)
    st.download_button("Download CSV", df_table.to_csv(index=False).encode("utf-8"), file_name="tracker.csv")
else:
    st.info("No rows to show yet. Check symbols above.")

if failed:
    st.warning("Failed to fetch or insufficient data for: " + ", ".join(failed))

st.caption("Tip: Add this page to your phone home screen (PWA shortcut) for quick access.")
