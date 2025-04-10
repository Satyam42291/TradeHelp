import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import streamlit as st
from datetime import date
import pandas as pd

start = '2000-01-01'
today = date.today().strftime("%Y-%m-%d")

st.title('📈 TradeHelp - Trading made easy')

popular_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
user_input = st.selectbox('Choose or enter a stock ticker:', popular_tickers, index=0)

n_years = st.slider("Years of Prediction:", 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    try:
        data = yf.download(ticker, start=start, end=today, auto_adjust=True)

        # Fix multi-index columns from yfinance >= 0.2.51
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() for col in data.columns.values]

        if data.empty:
            return None

        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error("❌ Failed to fetch data: " + str(e))
        return None

data_load_state = st.text("Loading data...")
data = load_data(user_input)

if data is None:
    st.error("📉 Could not fetch stock data. The ticker may be invalid, or Yahoo Finance may be facing issues.")
    st.stop()

data_load_state.text("✅ Data loaded successfully!")

st.subheader("Raw data")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open_AAPL'], name="Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close_AAPL'], name="Close"))
    fig.update_layout(title_text='📊 Time Series Data', 
                      xaxis_rangeslider_visible=True,
                      xaxis_title='Date',
                      yaxis_title='Price')
    st.plotly_chart(fig)

# Dynamically detect columns for selected ticker
close_col = [col for col in data.columns if 'Close' in col][0]
open_col = [col for col in data.columns if 'Open' in col][0]

plot_raw_data = lambda: st.plotly_chart(
    go.Figure([
        go.Scatter(x=data['Date'], y=data[open_col], name="Open"),
        go.Scatter(x=data['Date'], y=data[close_col], name="Close")
    ]).update_layout(
        title='📊 Time Series Data',
        xaxis_title='Date', yaxis_title='Price',
        xaxis_rangeslider_visible=True
    )
)

plot_raw_data()

# Moving Averages
ma100 = data[close_col].rolling(100).mean()
ma200 = data[close_col].rolling(200).mean()

fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=ma100, name='100MA', line=dict(color='red')))
fig.add_trace(go.Scatter(x=data['Date'], y=ma200, name='200MA', line=dict(color='green')))
fig.add_trace(go.Scatter(x=data['Date'], y=data[close_col], name='Close', line=dict(color='blue')))
fig.update_layout(
    xaxis_rangeslider_visible=True,
    width=800, 
    height=600,
    title='📉 Closing Price with 100MA and 200MA',
    xaxis_title='Date',
    yaxis_title='Price'
)
st.plotly_chart(fig)

# Forecasting with Prophet
df_train = data[['Date', close_col]]
df_train = df_train.rename(columns={"Date": "ds", close_col: "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('📈 Forecast Data')
st.write(forecast.tail())

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], mode='lines', name='Actual', line=dict(color='blue')))
fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='green')))
fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound', line=dict(color='red', dash='dash')))
fig1.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound', line=dict(color='orange', dash='dash')))

fig1.update_layout(
    xaxis_rangeslider_visible=True,
    width=800,
    height=600,
    title=f'📉 Forecast plot for {n_years} year(s)',
    xaxis_title='Date',
    yaxis_title='Price'
)
st.plotly_chart(fig1)

st.subheader("🔍 Forecast Components")
fig2 = m.plot_components(forecast)
st.write(fig2)

# FB PROPHET, RMSE=0.935556, calculation_time=0.659962893 approx.
