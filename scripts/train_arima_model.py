import warnings
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA

# Suppress warnings
warnings.filterwarnings("ignore")

def load_data(file_path="data/gold_prices.csv", anomaly_threshold=10000):
    """
    Load and prepare the cleaned gold price data, handling anomalies.
    """
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

    # Standardize column names to lowercase to match fetched data
    df.columns = df.columns.str.lower()

    # Ensure we are working with the correct column, typically 'close'
    if 'close' not in df.columns:
        raise KeyError("'close' column not found in the data!")

    # Remove rows where the 'close' price exceeds the anomaly threshold
    df = df[df['close'] < anomaly_threshold]
    return df[['close']]  # Only return the 'close' column as a DataFrame

def train_arima_model(data, forecast_days=5):
    """
    Train an ARIMA model and forecast future prices.
    """
    print("Finding optimal ARIMA parameters...")
    model_auto = auto_arima(data, seasonal=False, trace=True, stepwise=True)

    # Extract p, d, q values
    p, d, q = model_auto.order
    print(f"Optimal parameters: p={p}, d={d}, q={q}")
    
    # Train the ARIMA model with optimal parameters
    print("Training ARIMA model...")
    model = ARIMA(data, order=(p, d, q))
    model_fit = model.fit()

    # Forecast future values
    print("Forecasting future values...")
    forecast = model_fit.forecast(steps=forecast_days)
    
    return model_fit, forecast

def plot_forecast(data, model_fit, forecast):
    """
    Plot the historical data, in-sample predictions, and forecasts.
    """
    # Plot historical data and model fit
    plt.figure(figsize=(12, 6))
    plt.plot(data, label="Historical Gold Prices")
    plt.plot(model_fit.fittedvalues, color="orange", label="In-Sample Predictions")
    
    # Create future dates for the forecast
    forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=len(forecast), freq='D')
    forecast_series = pd.Series(forecast, index=forecast_index)
    
    # Plot forecasted data
    plt.plot(forecast_series, color="red", label="Forecast")
    plt.xlabel("Date")
    plt.ylabel("Gold Price (USD)")
    plt.legend()
    plt.title("Gold Price Prediction with ARIMA")
    plt.savefig('forecast_plot_cleaned.png')
    plt.show()
    

if __name__ == "__main__":
    # Load data
    data = load_data()

    # Train model and get forecast
    model_fit, forecast = train_arima_model(data)

    # Plot the results
    plot_forecast(data, model_fit, forecast)
