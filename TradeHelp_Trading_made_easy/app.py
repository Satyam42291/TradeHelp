
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
    # ensure datetime index if possible
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    return df

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

# ---------------- Resilient loader ----------------
@st.cache_data(show_spinner=False)
def load_data_resilient(ticker: str, start_date=start, end_date=today, max_retries=2):
    """
    Try (1) Ticker.history (2) yf.download (3) direct CSV fetch from Yahoo.
    Returns dict with either 'df' or 'error' plus diagnostics.
    """
    diagnostics = {"yfinance_version": getattr(yf, "__version__", "<unknown>"), "attempts": []}
    # 1) Try history()
    try:
        hist = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)
        if isinstance(hist, pd.DataFrame) and len(hist) > 0:
            df = hist.reset_index()
            df = flatten_multiindex_columns(df)
            diagnostics["attempts"].append({"method": "history", "rows": len(df)})
            return {"df": df, "source": "history", "diagnostics": diagnostics}
        diagnostics["attempts"].append({"method": "history", "rows": 0})
    except Exception as e:
        diagnostics["attempts"].append({"method": "history", "error": repr(e)})

    # 2) Try download()
    try:
        df2 = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False, threads=False)
        if isinstance(df2, pd.DataFrame) and len(df2) > 0:
            df2 = flatten_multiindex_columns(df2.reset_index())
            diagnostics["attempts"].append({"method": "download", "rows": len(df2)})
            return {"df": df2, "source": "download", "diagnostics": diagnostics}
        diagnostics["attempts"].append({"method": "download", "rows": 0})
    except Exception as e:
        diagnostics["attempts"].append({"method": "download", "error": repr(e)})

    # 3) Direct CSV fetch (bypass yfinance parsing)
    try:
        start_ts = int(pd.Timestamp(start_date).timestamp())
        end_ts = int(pd.Timestamp(end_date).timestamp())
        url = f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1={start_ts}&period2={end_ts}&interval=1d&events=history&includeAdjustedClose=true"
        headers = {"User-Agent": "Mozilla/5.0"}
        csv_text = None
        last_err = None
        for attempt in range(max_retries):
            try:
                r = requests.get(url, headers=headers, timeout=12)
                diagnostics["attempts"].append({"method": "direct_csv", "status_code": r.status_code, "len": len(r.text) if r.text else 0})
                if r.status_code == 200 and r.text and len(r.text) > 10:
                    csv_text = r.text
                    break
                last_err = f"status={r.status_code}, len={0 if r.text is None else len(r.text)}"
            except Exception as e:
                last_err = repr(e)
            time.sleep(0.5)
        if csv_text:
            df_csv = pd.read_csv(StringIO(csv_text), parse_dates=['Date'])
            df_csv = flatten_multiindex_columns(df_csv)
            diagnostics["attempts"].append({"method": "direct_csv_parsed", "rows": len(df_csv)})
            return {"df": df_csv, "source": "direct_csv", "diagnostics": diagnostics}
        diagnostics["attempts"].append({"method": "direct_csv", "error": last_err})
    except Exception as e:
        diagnostics["attempts"].append({"method": "direct_csv", "error": repr(e)})

    return {"error": "All attempts failed", "diagnostics": diagnostics}

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
    resp = load_data_resilient(user_input)
    if resp is None:
        st.error("Unexpected None from data loader.")
        st.stop()
    if "error" in resp:
        st.error("❌ Data load error: " + str(resp["error"]))
        st.write("Diagnostics:", resp.get("diagnostics", {}))
        st.stop()
    data = resp["df"]
    source = resp.get("source", "unknown")

st.write(f"Data source: {source}")
st.subheader("Raw data (tail)")
st.write(data.tail())

# ---------- Detect columns ----------
close_col = find_price_column(data.columns)
open_col = find_open_column(data.columns)

if close_col is None or open_col is None:
    st.error("Open/Close columns not found. Available columns: " + ", ".join([str(c) for c in data.columns]))
    st.stop()

# ---------- Plots ----------
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data[open_col], name="Open"))
fig.add_trace(go.Scatter(x=data['Date'], y=data[close_col], name="Close"))
fig.update_layout(title=f'📊 Time Series: {user_input if uploaded_file is None else "Uploaded data"}',
                  xaxis_title='Date', yaxis_title='Price',
                  xaxis_rangeslider_visible=True, width=1000, height=500)
st.plotly_chart(fig, use_container_width=True)

# Moving averages
ma100 = pd.to_numeric(data[close_col], errors='coerce').rolling(window=100, min_periods=1).mean()
ma200 = pd.to_numeric(data[close_col], errors='coerce').rolling(window=200, min_periods=1).mean()

fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=data['Date'], y=ma100, name='100MA'))
fig_ma.add_trace(go.Scatter(x=data['Date'], y=ma200, name='200MA'))
fig_ma.add_trace(go.Scatter(x=data['Date'], y=data[close_col], name='Close'))
fig_ma.update_layout(title='📉 Closing Price with 100MA and 200MA',
                     xaxis_title='Date', yaxis_title='Price',
                     xaxis_rangeslider_visible=True, width=1000, height=500)
st.plotly_chart(fig_ma, use_container_width=True)

# ---------- Prophet ----------
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
fig_forecast.update_layout(title=f'📉 Forecast plot for {n_years} year(s) ({user_input if uploaded_file is None else "Uploaded data"})',
                          xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=True,
                          width=1000, height=500)
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
