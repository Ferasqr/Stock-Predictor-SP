import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import matplotlib.pyplot as plt
import pmdarima as pm
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Constants
START = "2005-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Streamlit App Title
st.title('📈 Stock Forecast App')

# User Choice for Stock Selection
default_stocks = {'Apple (AAPL)': 'AAPL', 'Microsoft (MSFT)': 'MSFT', 'Google (GOOG)': 'GOOG'}
stock_choice = st.radio("Choose a stock or enter a custom ticker below:", list(default_stocks.keys()) + ['Custom'], horizontal=True)
if stock_choice == 'Custom':
    selected_stock = st.text_input('Enter a stock ticker for prediction:', '').upper()
else:
    selected_stock = default_stocks[stock_choice]

# If user input or selection is made, proceed with fetching and displaying data
if selected_stock:
    try:
        data = yf.Ticker(selected_stock).history(period='1d')
        if data.empty:
            st.error("No data found for the ticker! Please try another one.")
        else:
            n_years = st.slider('Years of Prediction:', 1, 7)
            period = n_years * 365
            forecast_model = st.selectbox('Select Forecast Model', ['Prophet', 'ARIMA', 'Exponential Smoothing', 'Random Forest'])

            @st.cache_data
            def load_data(ticker):
                """Load stock data from Yahoo Finance."""
                data = yf.download(ticker, START, TODAY)
                data.reset_index(inplace=True)
                return data

            # Load Data
            data_load_state = st.text('Loading data...')
            data = load_data(selected_stock)
            data_load_state.text('✅ Data Loaded Successfully!')
    except Exception as e:
   	 	st.error(f"Failed to retrieve data for {selected_stock}. Error: {str(e)}")
# Display Raw Data
st.subheader('📊 Raw Data')
st.write(data.tail())

# Function to plot raw data
def plot_raw_data(data):
    """Plot the raw stock data."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
    fig.layout.update(title_text='Time Series Data with Range Slider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
# Plot Raw Stock Data
st.subheader('📉 Raw Data Plot')
plot_raw_data(data)


# Prophet Model
if forecast_model == 'Prophet':
    st.subheader('📈 Prophet Model Forecast')
    df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)
    st.subheader('📈 Forecast Data')
    st.write(forecast.tail())
    st.subheader(f'📅 Forecast Plot for {n_years} Years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)
    st.subheader('📊 Forecast Components')
    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)

# ARIMA Model
elif forecast_model == 'ARIMA':
    st.subheader('📈 ARIMA Model Forecast')
    df_arima = data[['Date', 'Close']].set_index('Date')
    model = pm.auto_arima(df_arima, seasonal=False, stepwise=True)
    forecast, conf_int = model.predict(n_periods=period, return_conf_int=True)
    forecast_index = pd.date_range(df_arima.index[-1], periods=period + 1, freq='D')[1:]
    df_forecast = pd.DataFrame({'Date': forecast_index, 'Forecast': forecast})
    st.subheader('📈 Forecast Data')
    st.write(df_forecast.tail())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_arima.index, y=df_arima['Close'], name='Actual Close'))
    fig.add_trace(go.Scatter(x=df_forecast['Date'], y=df_forecast['Forecast'], name='Forecast', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df_forecast['Date'], y=conf_int[:, 0], fill=None, mode='lines', line_color='gray', name='Lower Conf. Int.'))
    fig.add_trace(go.Scatter(x=df_forecast['Date'], y=conf_int[:, 1], fill='tonexty', mode='lines', line_color='gray', name='Upper Conf. Int.'))
    fig.layout.update(title_text=f'ARIMA Forecast for {n_years} Years', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
# Exponential Smoothing Model
elif forecast_model == 'Exponential Smoothing':
    st.subheader('📈 Exponential Smoothing Forecast')
    df_es = data[['Date', 'Close']].set_index('Date')
    model = ExponentialSmoothing(df_es['Close'], trend='mul', seasonal='mul', seasonal_periods=365).fit()
    forecast = model.forecast(steps=period)
    forecast_index = pd.date_range(df_es.index[-1] + pd.Timedelta(days=1), periods=period, freq='D')
    df_forecast = pd.DataFrame({'Date': forecast_index, 'Forecast': forecast})
    st.subheader('📈 Forecast Data')
    st.write(df_forecast.tail())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_es.index, y=df_es['Close'], name='Actual Close'))
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast, name='Forecast', line=dict(color='orange')))
    fig.layout.update(title_text='Exponential Smoothing Forecast', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

# Random Forest Model
elif forecast_model == 'Random Forest':
    st.subheader('📈 Random Forest Model Forecast')
    df_rf = data[['Date', 'Close']].set_index('Date')
    X = np.array([i for i in range(len(df_rf))]).reshape(-1, 1)  # using days as features
    y = df_rf['Close'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    forecast = model.predict(np.array([i for i in range(len(X_train), len(X_train) + period)]).reshape(-1, 1))
    forecast_index = pd.date_range(df_rf.index[-1] + pd.Timedelta(days=1), periods=period, freq='D')
    df_forecast = pd.DataFrame({'Date': forecast_index, 'Forecast': forecast})

    st.subheader('📈 Forecast Data')
    st.write(df_forecast.tail())
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_rf.index, y=df_rf['Close'], name='Actual Close'))
    fig.add_trace(go.Scatter(x=forecast_index, y=forecast, name='Forecast', line=dict(color='orange')))
    fig.layout.update(title_text='Random Forest Forecast', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

