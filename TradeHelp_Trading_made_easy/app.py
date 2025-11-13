# app.py
"""
TradeHelp - Streamlit app (robust version)

- Handles yfinance MultiIndex columns (e.g., ('Close','AAPL')) by normalizing to strings.
- Includes diagnostic data-load attempts (yf.download threads True/False, Ticker.history).
- Shows helpful diagnostics in the UI when downloads fail.
- Safe column detection, input sanitization, and caching for downloads and forecasts.
- Uses Prophet for forecasting (may require cmdstan/cmdstanpy installation).
"""

import traceback
import urllib.request
import socket
from datetime import date

import pandas as pd
import streamlit as st
import yfinance as yf
from prophet import Prophet
from plotly import graph_objs as go
import matplotlib.pyplot as plt

# ---------------- Config ----------------
start = "2000-01-01"
today = date.today().strftime("%Y-%m-%d")

st.set_page_config(layout="wide", page_title="TradeHelp")
st.title("📈 TradeHelp - Trading made easy")

# ---------- UI inputs ----------
popular_tickers = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
choice = st.selectbox("Choose a stock ticker or type your own:",
                      popular_tickers + ["Type my own..."], index=0)
if choice == "Type my own...":
    user_input = st.text_input("Enter ticker (e.g. AAPL):", value="AAPL")
else:
    user_input = choice

# sanitize input
user_input = (user_input or "").strip().upper()

n_years = st.slider("Years of Prediction:", 1, 4)
period = n_years * 365  # forecast days

# ---------------- Helpers ----------------
def simple_http_check():
    """Quick check whether the Streamlit process can reach Yahoo Finance endpoints."""
    checks = []
    urls = [
        "https://query1.finance.yahoo.com",
        "https://query1.finance.yahoo.com/v7/finance/download/AAPL?period1=1609459200&period2=1703980800&interval=1d&events=history&includeAdjustedClose=true",
        "https://finance.yahoo.com/quote/AAPL",
    ]
    for u in urls:
        try:
            req = urllib.request.Request(u, method="HEAD")
            with urllib.request.urlopen(req, timeout=8) as resp:
                checks.append({"url": u, "status": resp.status, "reason": resp.reason})
        except Exception as e:
            checks.append({"url": u, "error": repr(e)})
    # DNS test
    try:
        ip = socket.gethostbyname("query1.finance.yahoo.com")
    except Exception as e:
        ip = f"DNS error: {repr(e)}"
    return {"http_checks": checks, "dns": ip}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns like ('Close','AAPL') to 'Close_AAPL' and ensure index/dtypes are sane.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if (x is not None and str(x) != "")]).strip()
            for col in df.columns.values
        ]
    else:
        # convert accidental tuple column names
        new_cols = []
        for c in df.columns:
            if isinstance(c, tuple):
                new_cols.append("_".join([str(x) for x in c if (x is not None and str(x) != "")]).strip())
            else:
                new_cols.append(c)
        df.columns = new_cols

    # try ensure datetime index if possible
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass

    return df


def find_price_column(cols, open_or_close="close"):
    """
    Find a best-match column for open/close among provided column names.
    open_or_close: 'close' or 'open'
    """
    keywords = {
        "close": ["adj close", "adj_close", "close", "close_"],
        "open": ["open", "open_"],
    }
    lc = [str(c).lower() for c in cols]
    for kw in keywords.get(open_or_close, []):
        for i, col in enumerate(lc):
            if kw in col:
                return cols[i]
    # fallback: any containing the word
    for i, col in enumerate(lc):
        if open_or_close in col:
            return cols[i]
    return None


@st.cache_data(show_spinner=False)
def load_data_debug(ticker: str):
    """
    Attempt multiple ways to fetch data and return a diagnostics dict.
    Returns:
      - {"df": dataframe, "source": "...", "diag": {...}} on success
      - {"error": "...", "diag": {...}} on failure
    """
    out = {"ticker_raw": repr(ticker), "yfinance_version": getattr(yf, "__version__", "unknown"), "attempts": []}
    try:
        # network checks
        out["net"] = simple_http_check()

        # attempt 1: yf.download with threads=True
        try:
            df1 = yf.download(ticker, start=start, end=today, auto_adjust=True, progress=False, threads=True)
            if isinstance(df1, pd.DataFrame):
                df1_norm = df1.copy()
                if isinstance(df1_norm.columns, pd.MultiIndex):
                    df1_norm.columns = [
                        "_".join([str(x) for x in col if (x is not None and str(x) != "")]).strip()
                        for col in df1_norm.columns.values
                    ]
                out["attempts"].append({
                    "method": "download(threads=True)",
                    "rows": len(df1_norm),
                    "cols": list(df1_norm.columns)[:10],
                    "head": df1_norm.head(1).to_dict()
                })
                if len(df1_norm) > 0:
                    return {"df": df1_norm.reset_index(), "source": "download_threads_true", "diag": out}
            else:
                out["attempts"].append({"method": "download(threads=True)", "result_type": str(type(df1)), "repr": repr(df1)[:300]})
        except Exception as e:
            out["attempts"].append({"method": "download(threads=True)", "error": str(e), "trace": traceback.format_exc()[:2000]})

        # attempt 2: yf.download with threads=False
        try:
            df2 = yf.download(ticker, start=start, end=today, auto_adjust=True, progress=False, threads=False)
            if isinstance(df2, pd.DataFrame):
                df2_norm = df2.copy()
                if isinstance(df2_norm.columns, pd.MultiIndex):
                    df2_norm.columns = [
                        "_".join([str(x) for x in col if (x is not None and str(x) != "")]).strip()
                        for col in df2_norm.columns.values
                    ]
                out["attempts"].append({
                    "method": "download(threads=False)",
                    "rows": len(df2_norm),
                    "cols": list(df2_norm.columns)[:10],
                    "head": df2_norm.head(1).to_dict()
                })
                if len(df2_norm) > 0:
                    return {"df": df2_norm.reset_index(), "source": "download_threads_false", "diag": out}
            else:
                out["attempts"].append({"method": "download(threads=False)", "result_type": str(type(df2)), "repr": repr(df2)[:300]})
        except Exception as e:
            out["attempts"].append({"method": "download(threads=False)", "error": str(e), "trace": traceback.format_exc()[:2000]})

        # attempt 3: Ticker.history(period="max")
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="max", auto_adjust=True)
            if isinstance(hist, pd.DataFrame):
                hist_norm = hist.copy()
                if isinstance(hist_norm.columns, pd.MultiIndex):
                    hist_norm.columns = [
                        "_".join([str(x) for x in col if (x is not None and str(x) != "")]).strip()
                        for col in hist_norm.columns.values
                    ]
                out["attempts"].append({
                    "method": "Ticker.history(period=max)",
                    "rows": len(hist_norm),
                    "cols": list(hist_norm.columns)[:10],
                    "head": hist_norm.head(1).to_dict()
                })
                if len(hist_norm) > 0:
                    return {"df": hist_norm.reset_index(), "source": "history_period_max", "diag": out}
            else:
                out["attempts"].append({"method": "Ticker.history(period=max)", "result_type": str(type(hist)), "repr": repr(hist)[:300]})
        except Exception as e:
            out["attempts"].append({"method": "Ticker.history(period=max)", "error": str(e), "trace": traceback.format_exc()[:2000]})

        # attempt 4: Ticker.history with explicit start/end
        try:
            t = yf.Ticker(ticker)
            hist2 = t.history(start=start, end=today, auto_adjust=True)
            if isinstance(hist2, pd.DataFrame):
                hist2_norm = hist2.copy()
                if isinstance(hist2_norm.columns, pd.MultiIndex):
                    hist2_norm.columns = [
                        "_".join([str(x) for x in col if (x is not None and str(x) != "")]).strip()
                        for col in hist2_norm.columns.values
                    ]
                out["attempts"].append({
                    "method": "Ticker.history(start,end)",
                    "rows": len(hist2_norm),
                    "cols": list(hist2_norm.columns)[:10],
                    "head": hist2_norm.head(1).to_dict()
                })
                if len(hist2_norm) > 0:
                    return {"df": hist2_norm.reset_index(), "source": "history_start_end", "diag": out}
            else:
                out["attempts"].append({"method": "Ticker.history(start,end)", "result_type": str(type(hist2)), "repr": repr(hist2)[:300]})
        except Exception as e:
            out["attempts"].append({"method": "Ticker.history(start,end)", "error": str(e), "trace": traceback.format_exc()[:2000]})

        # none returned data
        return {"error": "All attempts returned empty results", "diag": out}
    except Exception as e:
        return {"error": "Unexpected failure in load_data_debug: " + str(e), "trace": traceback.format_exc()[:2000], "diag": out}


# ---------------- Load data ----------------
st.text("Loading stock data...")
if not user_input:
    st.error("Please enter a valid ticker.")
    st.stop()

resp = load_data_debug(user_input)

if resp is None:
    st.error("Unexpected None from load_data_debug.")
    st.stop()

if "error" in resp:
    st.error("❌ Data load error: " + str(resp["error"]))
    # show diagnostics (helpful for debugging network/ticker issues)
    diag = resp.get("diag") or {}
    st.write("Diagnostics (short):")
    # show net check summary
    net = diag.get("net")
    if net:
        st.write("Network/DNS check:", net)
    attempts = diag.get("attempts", [])
    if attempts:
        st.write("Attempts summary:")
        for a in attempts:
            # keep output concise
            small = {k: a[k] for k in ("method", "rows", "cols") if k in a}
            st.write(small)
    # show full diag optionally collapsed
    with st.expander("Show full diagnostics"):
        st.json(diag)
    st.stop()

# success: we have a dataframe
data = resp["df"]
st.write(f"Data source: {resp.get('source','unknown')} — rows: {resp.get('rows')}")
st.subheader("Raw data (tail)")
st.write(data.tail())

# ---------- Detect open/close columns ----------
close_col = find_price_column(data.columns, open_or_close="close")
open_col = find_price_column(data.columns, open_or_close="open")

if close_col is None or open_col is None:
    st.error("Open/Close columns not found. Available columns: " + ", ".join([str(c) for c in data.columns]))
    st.stop()

# ---------- Plot raw time series ----------
fig = go.Figure()
fig.add_trace(go.Scatter(x=data["Date"], y=data[open_col], name="Open"))
fig.add_trace(go.Scatter(x=data["Date"], y=data[close_col], name="Close"))
fig.update_layout(
    title=f"📊 Time Series: {user_input}",
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=True,
    width=1000,
    height=500,
)
st.plotly_chart(fig, use_container_width=True)

# ---------- Moving averages ----------
ma100 = pd.to_numeric(data[close_col], errors="coerce").rolling(window=100, min_periods=1).mean()
ma200 = pd.to_numeric(data[close_col], errors="coerce").rolling(window=200, min_periods=1).mean()

fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=data["Date"], y=ma100, name="100MA"))
fig_ma.add_trace(go.Scatter(x=data["Date"], y=ma200, name="200MA"))
fig_ma.add_trace(go.Scatter(x=data["Date"], y=data[close_col], name="Close"))
fig_ma.update_layout(
    title="📉 Closing Price with 100MA and 200MA",
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=True,
    width=1000,
    height=500,
)
st.plotly_chart(fig_ma, use_container_width=True)

# ---------- Prepare data for Prophet ----------
df_train = data[["Date", close_col]].rename(columns={"Date": "ds", close_col: "y"})
df_train["ds"] = pd.to_datetime(df_train["ds"])
df_train["y"] = pd.to_numeric(df_train["y"], errors="coerce")
df_train = df_train.dropna()

if df_train.empty:
    st.error("Training dataframe is empty after sanitization.")
    st.stop()

# cache forecasting to avoid re-fitting on every interaction
@st.cache_data(show_spinner=True)
def train_and_forecast(df, periods: int):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq="D")
    forecast = model.predict(future)
    return model, forecast

with st.spinner("Training Prophet model and forecasting (this may take a moment)..."):
    try:
        model, forecast = train_and_forecast(df_train, period)
    except Exception as e:
        st.error("Prophet training/predict failed: " + str(e))
        # show a compact traceback for debugging
        st.text(traceback.format_exc().splitlines()[-6:])
        st.stop()

st.subheader("📈 Forecast Data (tail)")
st.write(forecast.tail())

# ---------- Plot forecast ----------
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=df_train["ds"], y=df_train["y"], mode="lines", name="Actual"))
fig_forecast.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast"))
fig_forecast.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], mode="lines", name="Upper Bound", line=dict(dash="dash")))
fig_forecast.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], mode="lines", name="Lower Bound", line=dict(dash="dash")))
fig_forecast.update_layout(
    title=f"📉 Forecast plot for {n_years} year(s) ({user_input})",
    xaxis_title="Date",
    yaxis_title="Price",
    xaxis_rangeslider_visible=True,
    width=1000,
    height=500,
)
st.plotly_chart(fig_forecast, use_container_width=True)

# ---------- Forecast components (matplotlib) ----------
st.subheader("🔍 Forecast Components")
try:
    fig_components = model.plot_components(forecast)
    st.pyplot(fig_components)
except Exception as e:
    st.write("Could not render Prophet components: " + str(e))

# ---------- Simple in-sample RMSE diagnostic ----------
try:
    merged = forecast[["ds", "yhat"]].merge(df_train[["ds", "y"]], on="ds", how="inner")
    if not merged.empty:
        mse = ((merged["y"] - merged["yhat"]) ** 2).mean()
        rmse = float(mse ** 0.5)
        st.write(f"Naive in-sample RMSE: `{rmse:.6f}` (diagnostic only)")
except Exception:
    pass
