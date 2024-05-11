# Stock-Predictor-SP
![55](https://github.com/Ferasqr/Stock-Predictor-SP/assets/93034515/07c3b285-5cb8-41c1-b8f9-f121a1a7cc3e)


The Stock-Predictor-SP is a powerful financial tool built using Streamlit, designed to predict future stock prices based on historical data from Yahoo Finance. This application offers a selection of forecasting models, including Prophet, ARIMA, Exponential Smoothing, and Random Forest, to provide insights into potential future stock price movements.

## Features

- **Stock Selection**: Choose from popular stocks like Apple, Microsoft, and Google, or enter a custom ticker.
- **Forecast Models**: Select from various models such as Prophet, ARIMA, Exponential Smoothing, or Random Forest.
- **Interactive Charts**: Visualize historical data and forecasts with interactive Plotly charts.
- **User-Friendly Interface**: Easy-to-navigate interface with horizontal radio button selection and dynamic input fields.

## Installation

To run the Stock Forecast App, you need to have Python installed on your system. If you don't have Python installed, download and install it from [python.org](https://www.python.org/downloads/).

Once Python is installed, follow these steps to set up the app:

1. **Clone the repository**:
git clone https://github.com/Ferasqr/Stock-Predictor-SP.git
cd Stock-Predictor-SP

3. **Install required libraries**:
pip install streamlit yfinance prophet pmdarima plotly matplotlib statsmodels sklearn

4. **Run the application**:
streamlit run main.py


## Usage

Upon launching the app, you will see the main interface where you can select a stock from the predefined list or enter a custom ticker. After selecting a stock and a forecast model, the app will display:

- Raw data of the selected stock.
- An interactive plot of historical stock prices.
- Predictions based on the selected model.
- Components and trends for certain models like Prophet.

To interact with the app:

1. **Choose a stock** from the radio options or enter a custom stock ticker.
2. **Select the number of years** you want the forecast to cover using the slider.
3. **Choose a forecast model** from the dropdown menu.
4. **View the results** which will include raw data, a plot of historical data, and forecast data.

## Contributions

Contributions are welcome. If you have suggestions to improve this app, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
