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

# Popular tickers and an option to type your own
popular_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
choice = st.selectbox('Choose a stock ticker or type your own:', popular_tickers + ['Type my own...'], index=0)
if choice == 'Type my own...':
    user_input = st.text_input("Enter ticker (e.g. AAPL):", value="AAPL").upper().strip()
else:
    user_input = choice

n_years = st.slider("Years of Prediction:", 1, 4)
period = n_years * 365  # forecast days

# ----------- Helpers / Safe utils -----------
def find_column_like(cols, keywords):
    """Return first column that contains any keyword (case-insensitive) or None."""
    lower_cols = [c.lower() for c in cols]
    for kw in keywords:
        kw_lower = kw.lower()
        for idx, c in enumerate(lower_cols):
            if kw_lower in c:
                return cols[idx]
    return None

@st.cache_data(show_spinner=False)
def load_data(ticker: str):
    """Download stock data and normalize column names for single vs multi-ticker outputs."""
    try:
        data = yf.download(ticker, start=start, end=today, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]
        data.reset_index(inplace=True)
        if data.empty:
            return None
        return data
    except Exception as e:
        # return an object that signals an error for clearer handling upstream
        return {"error": str(e)}

# ---------------- Load data ----------------
data_load_state = st.text("Loading data...")
raw = load_data(user_input)

if raw is None:
    st.error("📉 Could not fetch stock data. The ticker may be invalid or Yahoo Finance may be unreachable.")
    st.stop()
if isinstance(raw, dict) and raw.get("error"):
    st.error("❌ Failed to fetch data: " + raw["error"])
    st.stop()

data = raw
data_load_state.text("✅ Data loaded successfully!")

st.subheader("Raw data (last 5 rows)")
st.write(data.tail())

# --------- Detect columns safely ----------
close_col = find_column_like(data.columns, ['Close', 'Adj Close', 'close_'])
open_col = find_column_like(data.columns, ['Open', 'open_'])

if close_col is None or open_col is None:
    st.error("Required price columns (Open/Close) were not found in the downloaded data.")
    st.write("Available columns:", list(data.columns))
    st.stop()

# ---------- Plot raw time series ----------
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data[open_col], name="Open"))
fig.add_trace(go.Scatter(x=data['Date'], y=data[close_col], name="Close"))
fig.update_layout(
    title=f'📊 Time Series: {user_input}',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=True,
    width=1000, height=500
)
st.plotly_chart(fig, use_container_width=True)

# ---------- Moving averages ----------
ma100 = data[close_col].rolling(window=100, min_periods=1).mean()
ma200 = data[close_col].rolling(window=200, min_periods=1).mean()

fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=data['Date'], y=ma100, name='100MA'))
fig_ma.add_trace(go.Scatter(x=data['Date'], y=ma200, name='200MA'))
fig_ma.add_trace(go.Scatter(x=data['Date'], y=data[close_col], name='Close'))
fig_ma.update_layout(
    title='📉 Closing Price with 100MA and 200MA',
    xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=True,
    width=1000, height=500
)
st.plotly_chart(fig_ma, use_container_width=True)

# ---------- Prepare data for Prophet ----------
df_train = data[['Date', close_col]].rename(columns={'Date': 'ds', close_col: 'y'})
df_train['ds'] = pd.to_datetime(df_train['ds'])
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
df_train = df_train.dropna()

if df_train.empty:
    st.error("Training dataframe is empty after sanitization.")
    st.stop()

# Cache training + forecast to avoid retraining on every interaction
@st.cache_data(show_spinner=True)
def train_and_forecast(df, periods: int):
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods, freq='D')
    forecast = model.predict(future)
    return model, forecast

with st.spinner("Training forecasting model (this may take a moment)..."):
    try:
        model, forecast = train_and_forecast(df_train, period)
    except Exception as e:
        st.error("Prophet training/prediction failed: " + str(e))
        st.stop()

st.subheader('📈 Forecast Data (tail)')
st.write(forecast.tail())

# ---------- Plot forecast vs actual ----------
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], mode='lines', name='Actual'))
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dash')))
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dash')))
fig_forecast.update_layout(
    title=f'📉 Forecast plot for {n_years} year(s) ({user_input})',
    xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=True,
    width=1000, height=500
)
st.plotly_chart(fig_forecast, use_container_width=True)

# ---------- Forecast components  (matplotlib) ----------
st.subheader("🔍 Forecast Components")
try:
    fig_components = model.plot_components(forecast)
    st.pyplot(fig_components)
except Exception as e:
    st.write("Could not render Prophet components:", str(e))

# ---------- Optional: simple performance metric ----------
# NOTE: This is a naive in-sample metric (for quick diagnostics only)
try:
    # Compare model yhat on training dates to actuals (inner join by ds)
    merged = forecast[['ds', 'yhat']].merge(df_train[['ds', 'y']], on='ds', how='inner')
    if not merged.empty:
        mse = ((merged['y'] - merged['yhat']) ** 2).mean()
        rmse = float(mse ** 0.5)
        st.write(f"Naive in-sample RMSE: `{rmse:.6f}` (for diagnostic purposes only)")
except Exception:
    pass
