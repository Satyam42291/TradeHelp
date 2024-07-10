import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import streamlit as st
from datetime import date


start = '2000-01-01'
today = date.today().strftime("%Y-%m-%d")
# today = '2022-12-31'

st.title('TradeHelp - Trading made easy ;)')

user_input=st.text_input('Enter Stock Ticker (as per Yahoo finance)', 'AAPL')

n_years= st.slider("Years of Prediction:", 1, 5)
period= n_years*365

@st.cache_data
def load_data(ticker):
    data= yf.download(ticker, start, today)
    data.reset_index(inplace=True)
    return data

data_load_state= st.text("load data...")
data= load_data(user_input)
data_load_state.text("loading data... done!")

st.subheader("Raw data")
st.write(data.tail())

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data',
                    xaxis_rangeslider_visible=True,
                    xaxis_title='Time',
                    yaxis_title='Price'
                    )
	st.plotly_chart(fig)

plot_raw_data()

# st.subheader('Closing Price vs Time chart with 100MA and 200MA')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=ma100, mode='lines', name='100MA', line=dict(color='red')))
fig.add_trace(go.Scatter(x=data['Date'], y=ma200, mode='lines', name='200MA', line=dict(color='green')))
fig.add_trace(go.Scatter(x=data['Date'], y=data.Close, mode='lines', name='Closing Price', line=dict(color='blue')))
fig.update_layout(
    xaxis_rangeslider_visible=True,
    width=800, 
    height=600,
    title='Closing Price vs Time chart with 100MA and 200MA',
    xaxis_title='Time',
    yaxis_title='Price'
)
st.plotly_chart(fig)

#Here, forecasting begins..
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

forecast['yhat_upper'] = forecast['yhat_upper']
forecast['yhat_lower'] = forecast['yhat_lower']

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=df_train['ds'], 
    y=df_train['y'], 
    mode='lines', 
    name='Actual', 
    line=dict(color='blue')
))

fig1.add_trace(go.Scatter(
    x=forecast['ds'], 
    y=forecast['yhat'], 
    mode='lines', 
    name='Forecast', 
    line=dict(color='green')
))

fig1.add_trace(go.Scatter(
    x=forecast['ds'], 
    y=forecast['yhat_upper'], 
    mode='lines', 
    name='Upper Bound', 
    line=dict(color='red', dash='dash')
))

fig1.add_trace(go.Scatter(
    x=forecast['ds'], 
    y=forecast['yhat_lower'], 
    mode='lines', 
    name='Lower Bound', 
    line=dict(color='orange', dash='dash')
))

fig1.update_layout(
    xaxis_rangeslider_visible=True,
    width=800, 
    height=600,
    title=f'Forecast plot for {n_years} years',
    xaxis_title='Date',
    yaxis_title='Price'
)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)


# FB PROPHET, RMSE=0.935556, calculation_time=0.659962893 approx.
