# import yfinance as yf
# from prophet import Prophet
# from prophet.plot import plot_plotly
# from plotly import graph_objs as go
# import streamlit as st
# from datetime import date
# import pandas as pd

# start = '2000-01-01'
# today = date.today().strftime("%Y-%m-%d")

# st.title('📈 TradeHelp - Trading made easy')

# popular_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
# user_input = st.selectbox('Choose or enter a stock ticker:', popular_tickers, index=0)

# n_years = st.slider("Years of Prediction:", 1, 4)
# period = n_years * 365

# @st.cache_data
# def load_data(ticker):
#     try:
#         data = yf.download(ticker, start=start, end=today, auto_adjust=True)

#         # Fix multi-index columns from yfinance >= 0.2.51
#         if isinstance(data.columns, pd.MultiIndex):
#             data.columns = ['_'.join(col).strip() for col in data.columns.values]

#         if data.empty:
#             return None

#         data.reset_index(inplace=True)
#         return data
#     except Exception as e:
#         st.error("❌ Failed to fetch data: " + str(e))
#         return None

# data_load_state = st.text("Loading data...")
# data = load_data(user_input)

# if data is None:
#     st.error("📉 Could not fetch stock data. The ticker may be invalid, or Yahoo Finance may be facing issues.")
#     st.stop()

# data_load_state.text("✅ Data loaded successfully!")

# st.subheader("Raw data")
# st.write(data.tail())

# def plot_raw_data():
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=data['Date'], y=data['Open_AAPL'], name="Open"))
#     fig.add_trace(go.Scatter(x=data['Date'], y=data['Close_AAPL'], name="Close"))
#     fig.update_layout(title_text='📊 Time Series Data', 
#                       xaxis_rangeslider_visible=True,
#                       xaxis_title='Date',
#                       yaxis_title='Price')
#     st.plotly_chart(fig)

# # Dynamically detect columns for selected ticker
# close_col = [col for col in data.columns if 'Close' in col][0]
# open_col = [col for col in data.columns if 'Open' in col][0]

# plot_raw_data = lambda: st.plotly_chart(
#     go.Figure([
#         go.Scatter(x=data['Date'], y=data[open_col], name="Open"),
#         go.Scatter(x=data['Date'], y=data[close_col], name="Close")
#     ]).update_layout(
#         title='📊 Time Series Data',
#         xaxis_title='Date', yaxis_title='Price',
#         xaxis_rangeslider_visible=True
#     )
# )

# plot_raw_data()

# # Moving Averages
# ma100 = data[close_col].rolling(100).mean()
# ma200 = data[close_col].rolling(200).mean()

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=data['Date'], y=ma100, name='100MA', line=dict(color='red')))
# fig.add_trace(go.Scatter(x=data['Date'], y=ma200, name='200MA', line=dict(color='green')))
# fig.add_trace(go.Scatter(x=data['Date'], y=data[close_col], name='Close', line=dict(color='blue')))
# fig.update_layout(
#     xaxis_rangeslider_visible=True,
#     width=800, 
#     height=600,
#     title='📉 Closing Price with 100MA and 200MA',
#     xaxis_title='Date',
#     yaxis_title='Price'
# )
# st.plotly_chart(fig)

# # Forecasting with Prophet
# df_train = data[['Date', close_col]]
# df_train = df_train.rename(columns={"Date": "ds", close_col: "y"})

# m = Prophet()
# m.fit(df_train)
# future = m.make_future_dataframe(periods=period)
# forecast = m.predict(future)

# st.subheader('📈 Forecast Data')
# st.write(forecast.tail())

# fig1 = go.Figure()
# fig1.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], mode='lines', name='Actual', line=dict(color='blue')))
# fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='green')))
# fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(color='red', dash='dash')))
# fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(color='orange', dash='dash')))

# fig1.update_layout(
#     xaxis_rangeslider_visible=True,
#     width=800,
#     height=600,
#     title=f'📉 Forecast plot for {n_years} year(s)',
#     xaxis_title='Date',
#     yaxis_title='Price'
# )
# st.plotly_chart(fig1)

# st.subheader("🔍 Forecast Components")
# fig2 = m.plot_components(forecast)
# st.write(fig2)

# FB PROPHET, RMSE=0.935556, calculation_time=0.659962893 approx.














# app.py
import yfinance as yf
from prophet import Prophet
#from prophet.plot import plot_plotly  # (optional) not used here
from plotly import graph_objs as go
import streamlit as st
from datetime import date
import pandas as pd

# --- Config ---
start = '2000-01-01'
today = date.today().strftime("%Y-%m-%d")

st.set_page_config(layout="wide", page_title="TradeHelp")
st.title('📈 TradeHelp - Trading made easy')

popular_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
user_input = st.selectbox('Choose or enter a stock ticker:', popular_tickers, index=0, 
                          options=popular_tickers + ["Type your own..."])

# allow typing a custom ticker if user selected placeholder
if user_input == "Type your own...":
    user_input = st.text_input("Enter ticker (e.g. AAPL):", value="AAPL").upper().strip()

n_years = st.slider("Years of Prediction:", 1, 4)
period = n_years * 365  # days

@st.cache_data(show_spinner=False)
def load_data(ticker: str):
    try:
        data = yf.download(ticker, start=start, end=today, auto_adjust=True)
        # Fix multi-index columns from yfinance >= 0.2.51
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]
        data.reset_index(inplace=True)
        if data.empty:
            return None
        return data
    except Exception as e:
        # return exception message for better debugging upstream
        return {"error": str(e)}

data_load_state = st.text("Loading data...")
data_or_err = load_data(user_input)

# handle errors
if data_or_err is None:
    st.error("📉 Could not fetch stock data. The ticker may be invalid, or Yahoo Finance may be facing issues.")
    st.stop()
if isinstance(data_or_err, dict) and data_or_err.get("error"):
    st.error("❌ Failed to fetch data: " + data_or_err["error"])
    st.stop()

data = data_or_err
data_load_state.text("✅ Data loaded successfully!")

st.subheader("Raw data (last rows)")
st.write(data.tail())

# Robust detection of open/close columns
def find_column_like(cols, keywords):
    for kw in keywords:
        for c in cols:
            if kw.lower() in c.lower():
                return c
    return None

close_col = find_column_like(data.columns, ['Close', 'Adj Close', 'Close_'])
open_col = find_column_like(data.columns, ['Open', 'Open_'])

if close_col is None or open_col is None:
    st.error("Required price columns (Open/Close) were not found in the downloaded data.")
    st.write("Found columns:", list(data.columns))
    st.stop()

# Time series plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data[open_col], name="Open"))
fig.add_trace(go.Scatter(x=data['Date'], y=data[close_col], name="Close"))
fig.update_layout(
    title='📊 Time Series Data',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=True,
    width=900, height=500
)
st.plotly_chart(fig, use_container_width=True)

# Moving averages (handle NaN safely)
ma100 = data[close_col].rolling(window=100, min_periods=1).mean()
ma200 = data[close_col].rolling(window=200, min_periods=1).mean()

fig_ma = go.Figure()
fig_ma.add_trace(go.Scatter(x=data['Date'], y=ma100, name='100MA'))
fig_ma.add_trace(go.Scatter(x=data['Date'], y=ma200, name='200MA'))
fig_ma.add_trace(go.Scatter(x=data['Date'], y=data[close_col], name='Close'))
fig_ma.update_layout(
    title='📉 Closing Price with 100MA and 200MA',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=True,
    width=900, height=500
)
st.plotly_chart(fig_ma, use_container_width=True)

# Prepare data for Prophet
df_train = data[['Date', close_col]].rename(columns={'Date': 'ds', close_col: 'y'})
# ensure ds is datetime and y is numeric
df_train['ds'] = pd.to_datetime(df_train['ds'])
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
df_train = df_train.dropna()

if df_train.empty:
    st.error("Training dataframe for Prophet is empty after sanitization.")
    st.stop()

# Fit & forecast (cache the forecast result by ticker+period to avoid refitting frequently)
@st.cache_data(show_spinner=True)
def train_and_forecast(df, periods: int):
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=periods, freq='D')
    forecast = m.predict(future)
    return m, forecast

with st.spinner("Training Prophet model and forecasting..."):
    try:
        model, forecast = train_and_forecast(df_train, period)
    except Exception as e:
        st.error("❌ Prophet training / prediction failed: " + str(e))
        st.stop()

st.subheader('📈 Forecast Data (tail)')
st.write(forecast.tail())

# Plot actual vs forecast
fig_forecast = go.Figure()
fig_forecast.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], mode='lines', name='Actual'))
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(dash='dash')))
fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(dash='dash')))
fig_forecast.update_layout(
    title=f'📉 Forecast plot for {n_years} year(s)',
    xaxis_title='Date',
    yaxis_title='Price',
    xaxis_rangeslider_visible=True,
    width=900, height=500
)
st.plotly_chart(fig_forecast, use_container_width=True)

# Forecast components (matplotlib) - show explicitly with st.pyplot
st.subheader("🔍 Forecast Components")
try:
    fig_components = model.plot_components(forecast)
    import matplotlib.pyplot as plt
    st.pyplot(fig_components)
except Exception as e:
    st.write("Could not render Prophet components: ", str(e))

