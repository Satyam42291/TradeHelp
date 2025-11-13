# app.py
"""
TradeHelp — Streamlit app using yf.Ticker.history() as primary fetch method (per your reference).
Requirements (recommended):
streamlit, pandas, plotly, yfinance (recommended 0.2.25 if you had parsing issues),
prophet, cmdstanpy, matplotlib, requests
"""

import time
from io import StringIO
from datetime import date
import requests

import yfinance as yf
from prophet import Prophet
from plotly import graph_objs as go
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Config ----------------
start = '2000-01-01'
today = date.today().strftime("%Y-%m-%d")

st.set_page_config(layout="wide", page_title="TradeHelp")
st.title('📈 TradeHelp - Trading made easy')

# --- UI: ticker selection + file upload fallback ---
popular_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
options = popular_tickers + ['Type my own...']
choice = st.selectbox('Choose a stock ticker or type your own:', options, index=0)

if choice == 'Type my own...':
    user_input = st.text_input("Enter ticker (e.g. AAPL):", value="AAPL").upper().strip()
else:
    user_input = choice

st.write("Or upload a CSV with a `Date` column (used as fallback):")
uploaded_file = st.file_uploader("Upload historical CSV (optional)", type=["csv"])

n_years = st.slider("Years of Prediction:", 1, 4)
period = n_years * 365  # days

# ---------------- Helpers ----------------
def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert MultiIndex columns to single-string names like 'Close_AAPL'."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(x) for x in col if (x is not None and str(x) != '')]).strip() for col in df.columns.values]
    else:
        # convert accidental tuple columns
        new_cols = []
        for c in df.columns:
            if isinstance(c, tuple):
                new_cols.append('_'.join([str(x) for x in c if (x is not None and str(x) != '')]).strip())
            else:
                new_cols.append(c)
        df.columns = new_cols
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    return df

def load_data_history_style(ticker: str):
    """
    Fetch using yf.Ticker(ticker).history(...) as in your reference code.
    Returns dict { 'df': DataFrame } or { 'error': str }.
    DataFrame has at least columns: ['Date', 'Close'] where Date is datetime.
    """
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(start=start, end=today, actions=False, auto_adjust=True)
        # If empty, return error
        if hist is None or len(hist) == 0:
            return {"error": f"history() returned empty for ticker {ticker}"}
        # flatten columns if needed
        hist = flatten_multiindex_columns(hist)
        # Convert index to Date column (keep datetime for Prophet later)
        # Reference code converted to date; we'll keep datetime (safer for plotting/Prophet)
        hist = hist.reset_index()
        # Ensure there is a Date column
        if 'Date' not in hist.columns and hist.index.name is None:
            hist['Date'] = hist.index
        # Keep Close (and Open if available for plotting)
        if 'Close' not in hist.columns and 'Close_AAPL' in hist.columns:
            # fallback if flattened column naming used ticker suffix
            close_cols = [c for c in hist.columns if 'close' in c.lower()]
            if close_cols:
                hist['Close'] = hist[close_cols[0]]
        # Convert Date to datetime (already likely so)
        hist['Date'] = pd.to_datetime(hist['Date'])
        # Select only necessary columns (but keep Open if present)
        cols_keep = ['Date']
        if 'Open' in hist.columns:
            cols_keep.append('Open')
        cols_keep.append('Close')
        # Filter columns that actually exist
        cols_keep = [c for c in cols_keep if c in hist.columns]
        final_df = hist[cols_keep].copy()
        # Match reference behavior: remove index name and columns name
        final_df.index.name = None
        final_df.columns.name = None
        return {"df": final_df}
    except Exception as e:
        return {"error": str(e)}

def find_price_column(cols, preferred_keywords=('close', 'adj close', 'adj_close', 'close_')):
    lc = [str(c).lower() for c in cols]
    for kw in preferred_keywords:
        for i, c in enumerate(lc):
            if kw in c:
                return cols[i]
    for i, c in enumerate(lc):
        if 'close' in c:
            return cols[i]
    return None

def find_open_column(cols, preferred_keywords=('open', 'open_')):
    lc = [str(c).lower() for c in cols]
    for kw in preferred_keywords:
        for i, c in enumerate(lc):
            if kw in c:
                return cols[i]
    for i, c in enumerate(lc):
        if 'open' in c:
            return cols[i]
    return None

# ---------------- Main flow ----------------
st.text("Loading data...")

# If user uploaded CSV, use it (highest priority)
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file, parse_dates=['Date'])
        data = flatten_multiindex_columns(data)
        source = "uploaded_csv"
        st.success("Loaded data from uploaded CSV.")
    except Exception as e:
        st.error("Failed to parse uploaded CSV: " + str(e))
        st.stop()
else:
    resp = load_data_history_style(user_input)
    if resp is None:
        st.error("Unexpected None from data loader.")
        st.stop()
    if "error" in resp:
        # As fallback, attempt a resilient loader (direct CSV) — minimal attempt here
        st.warning("history() failed: " + resp["error"] + " — attempting yf.download() fallback...")
        # Try download briefly
        try:
            df2 = yf.download(user_input, start=start, end=today, auto_adjust=True, progress=False, threads=False)
            if df2 is not None and len(df2) > 0:
                df2 = flatten_multiindex_columns(df2.reset_index())
                # produce DataFrame similar to reference
                if 'Date' not in df2.columns:
                    df2['Date'] = pd.to_datetime(df2['Date'] if 'Date' in df2.columns else df2.index)
                final_df = df2[['Date'] + ([c for c in ['Open','Close'] if c in df2.columns])]
                final_df.index.name = None
                final_df.columns.name = None
                data = final_df
                source = "download_fallback"
                st.success("yf.download() fallback succeeded.")
            else:
                st.error("yf.download() fallback returned empty.")
                st.stop()
        except Exception as e:
            st.error("Fallback download() failed: " + str(e))
            st.stop()
    else:
        data = resp["df"]
        source = "history"

st.write(f"Data source: {source}")
st.subheader("Raw data (tail)")
st.write(data.tail())

# ---------- Detect columns ----------
close_col = find_price_column(data.columns)
open_col = find_open_column(data.columns)

if close_col is None:
    st.error("Close column not found. Available columns: " + ", ".join([str(c) for c in data.columns]))
    st.stop()

# ---------- Plots ----------
fig = go.Figure()
# Plot Open if available (reference kept Open in their code)
if open_col is not None:
    fig.add_trace(go.Scatter(x=data['Date'], y=data[open_col], name="Open"))
fig.add_trace(go.Scatter(x=data['Date'], y=data[close_col], name="Close"))
fig.update_layout(title=f'📊 Time Series: {user_input}', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=True, width=1000, height=500)
st.plotly_chart(fig, use_container_width=True)

# Moving averages (based on Close)
ma100 = pd.to_numeric(data[close_col], errors='coerce').rolling(window=100, min_periods=1).mean()
ma200 = pd.to_numeric(data[close_col], errors='coerce').rolling(window=200, min_periods=1).mean()

fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=data['Date'], y=ma100, name='100MA'))
fig_ma.add_trace(go.Scatter(x=data['Date'], y=ma200, name='200MA'))
fig_ma.add_trace(go.Scatter(x=data['Date'], y=data[close_col], name='Close'))
fig_ma.update_layout(title='📉 Closing Price with 100MA and 200MA', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=True, width=1000, height=500)
st.plotly_chart(fig_ma, use_container_width=True)

# ---------- Prophet ----------
# Prepare DataFrame similar to reference but with datetime for Prophet
df_train = data[['Date', close_col]].rename(columns={'Date': 'ds', close_col: 'y'})
df_train['ds'] = pd.to_datetime(df_train['ds'])
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
df_train = df_train.dropna()
if df_train.empty:
    st.error("Training dataframe is empty after sanitization.")
    st.stop()

@st.cache_data(show_spinner=True)
def train_and_forecast(df, periods: int):
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=periods, freq='D')
    forecast = m.predict(future)
    return m, forecast

with st.spinner("Training forecasting model (may take a moment)..."):
    try:
        model, forecast = train_and_forecast(df_train, period)
    except Exception as e:
        st.error("Prophet training/predict failed: " + str(e))
        st.stop()

st.subheader('📈 Forecast Data (tail)')
st.write(forecast.tail())

fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], mode='lines', name='Actual'))
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dash')))
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dash')))
fig_forecast.update_layout(title=f'📉 Forecast plot for {n_years} year(s) ({user_input if uploaded_file is None else "Uploaded data"})', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=True, width=1000, height=500)
st.plotly_chart(fig_forecast, use_container_width=True)

st.subheader("🔍 Forecast Components")
try:
    fig_comp = model.plot_components(forecast)
    st.pyplot(fig_comp)
except Exception as e:
    st.write("Could not render components:", str(e))

# ---------- Naive in-sample RMSE (diagnostic) ----------
try:
    merged = forecast[['ds', 'yhat']].merge(df_train[['ds', 'y']], on='ds', how='inner')
    if not merged.empty:
        mse = ((merged['y'] - merged['yhat']) ** 2).mean()
        rmse = float(mse ** 0.5)
        st.write(f"Naive in-sample RMSE: `{rmse:.6f}` (diagnostic only)")
except Exception:
    pass

# End of file
