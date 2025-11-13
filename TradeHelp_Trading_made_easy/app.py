# app.py
import yfinance as yf
from prophet import Prophet
from plotly import graph_objs as go
import streamlit as st
from datetime import date
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Config ----------------
start = '2000-01-01'
today = date.today().strftime("%Y-%m-%d")

st.set_page_config(layout="wide", page_title="TradeHelp")
st.title('📈 TradeHelp - Trading made easy')

popular_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
choice = st.selectbox('Choose a stock ticker or type your own:', popular_tickers + ['Type my own...'], index=0)
if choice == 'Type my own...':
    user_input = st.text_input("Enter ticker (e.g. AAPL):", value="AAPL").upper().strip()
else:
    user_input = choice

n_years = st.slider("Years of Prediction:", 1, 4)
period = n_years * 365  # days

# ------------- Helpers ----------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Turn MultiIndex columns like ('Close','AAPL') into 'Close_AAPL' strings.
       Also convert pandas PeriodIndex or other index types to DatetimeIndex if needed."""
    # If MultiIndex columns -> flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(x) for x in col if (x is not None and str(x) != '')]).strip() for col in df.columns.values]
    else:
        # convert any column names that are tuples accidentally
        new_cols = []
        for c in df.columns:
            if isinstance(c, tuple):
                new_cols.append('_'.join([str(x) for x in c if (x is not None and str(x) != '')]).strip())
            else:
                new_cols.append(c)
        df.columns = new_cols

    # Ensure index is datetime (yfinance usually has Date index)
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass

    return df

@st.cache_data(show_spinner=False)
def load_data(ticker: str):
    """Download and normalize data. Returns dict with 'df' or 'error' for diagnostics."""
    try:
        df = yf.download(ticker, start=start, end=today, auto_adjust=True, progress=False, threads=False)
        df = normalize_columns(df)
        if df is None or len(df) == 0:
            # try fallback .history()
            hist = yf.Ticker(ticker).history(start=start, end=today, auto_adjust=True)
            if hist is None or len(hist) == 0:
                return {"error": "Both yf.download() and Ticker.history() returned empty dataframes."}
            hist = normalize_columns(hist.reset_index())
            return {"df": hist, "source": "history_fallback", "rows": len(hist), "cols": list(hist.columns)}
        # reset_index to make Date a column (consistent shape)
        df_reset = df.reset_index()
        return {"df": df_reset, "source": "download", "rows": len(df_reset), "cols": list(df_reset.columns)}
    except Exception as e:
        return {"error": str(e)}

def find_price_column(cols, preferred_keywords=('close', 'adj close', 'adj_close', 'close_')):
    """Return first column name that matches any keyword (case-insensitive)."""
    lc = [c.lower() for c in cols]
    for kw in preferred_keywords:
        for i, c in enumerate(lc):
            if kw in c:
                return cols[i]
    # fallback: find any column containing 'close' or 'open'
    for i, c in enumerate(lc):
        if 'close' in c:
            return cols[i]
    return None

def find_open_column(cols, preferred_keywords=('open', 'open_')):
    lc = [c.lower() for c in cols]
    for kw in preferred_keywords:
        for i, c in enumerate(lc):
            if kw in c:
                return cols[i]
    for i, c in enumerate(lc):
        if 'open' in c:
            return cols[i]
    return None

# ---------------- Load data ----------------
st.text("Loading data...")
resp = load_data(user_input)

if resp is None:
    st.error("Unexpected None from load_data.")
    st.stop()

if "error" in resp:
    st.error("❌ Data load error: " + str(resp["error"]))
    if 'cols' in resp:
        st.write("Diagnostics:", {k:v for k,v in resp.items() if k != "error"})
    st.stop()

data = resp["df"]
st.write(f"Data source: {resp.get('source','unknown')} — rows: {resp.get('rows')}")
st.subheader("Raw data (tail)")
st.write(data.tail())

# ---------- Detect columns ----------
close_col = find_price_column(data.columns)
open_col = find_open_column(data.columns)

if close_col is None or open_col is None:
    st.error("Open/Close columns not found. Available columns: " + ", ".join(data.columns.astype(str)))
    st.stop()

# ---------- Plots ----------
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data[open_col], name="Open"))
fig.add_trace(go.Scatter(x=data['Date'], y=data[close_col], name="Close"))
fig.update_layout(title=f'📊 Time Series: {user_input}', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=True, width=1000, height=500)
st.plotly_chart(fig, use_container_width=True)

# Moving averages
ma100 = pd.to_numeric(data[close_col], errors='coerce').rolling(window=100, min_periods=1).mean()
ma200 = pd.to_numeric(data[close_col], errors='coerce').rolling(window=200, min_periods=1).mean()

fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=data['Date'], y=ma100, name='100MA'))
fig_ma.add_trace(go.Scatter(x=data['Date'], y=ma200, name='200MA'))
fig_ma.add_trace(go.Scatter(x=data['Date'], y=data[close_col], name='Close'))
fig_ma.update_layout(title='📉 Closing Price with 100MA and 200MA', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=True, width=1000, height=500)
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
fig_forecast.update_layout(title=f'📉 Forecast plot for {n_years} year(s) ({user_input})', xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=True, width=1000, height=500)
st.plotly_chart(fig_forecast, use_container_width=True)

st.subheader("🔍 Forecast Components")
try:
    fig_comp = model.plot_components(forecast)
    st.pyplot(fig_comp)
except Exception as e:
    st.write("Could not render components:", str(e))
