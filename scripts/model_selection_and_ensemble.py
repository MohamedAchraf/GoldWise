import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
from datetime import timedelta
import matplotlib

matplotlib.use('Agg')  # Use a non-interactive backend for servers

warnings.filterwarnings("ignore")


def preprocess_data(data):
    data = data.dropna()  # Remove any NaN values
    if isinstance(data, pd.DataFrame):
        data = data.squeeze()  # Convert to Series if one-column DataFrame
    return data


def detect_anomalies(data, z_thresh=3):
    """
    Detect anomalies using Z-score. Any data point beyond z_thresh standard deviations from the mean is considered an anomaly.
    """
    data_mean = np.mean(data)
    data_std = np.std(data)
    
    # Z-score for each data point
    z_scores = (data - data_mean) / data_std
    
    # Filter out anomalies
    filtered_data = data[(np.abs(z_scores) < z_thresh)]
    
    print(f"Detected and removed {len(data) - len(filtered_data)} anomalies from the data.")
    
    return filtered_data


def run_arima(data):
    print("\n================ Running ARIMA Model ================\n")
    try:
        print("Finding optimal ARIMA parameters...")
        model_auto = auto_arima(data, seasonal=False, trace=True)
        p, d, q = model_auto.order
        print(f"Optimal ARIMA parameters: (p={p}, d={d}, q={q})")
        
        model = ARIMA(data, order=(p, d, q))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=5).values  # Convert forecast to numpy array
        mse = mean_squared_error(data[-5:], forecast)
        print("ARIMA model training and forecasting completed.")
        return forecast, mse
    except Exception as e:
        print(f"ARIMA failed: {e}")
        return [np.nan] * 5, float('inf')


def run_prophet(data):
    print("\n================ Running Prophet Model ================\n")
    try:
        df = data.reset_index().rename(columns={"Date": "ds", "Close": "y"})
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=5)
        forecast = model.predict(future)["yhat"].iloc[-5:].values
        mse = mean_squared_error(data[-5:], forecast)
        print("Prophet model training and forecasting completed.")
        return forecast, mse
    except Exception as e:
        print(f"Prophet failed: {e}")
        return [np.nan] * 5, float('inf')


def run_ets(data, trend='add', seasonal='add', seasonal_periods=5):
    print("\n================ Running ETS Model ================\n")
    try:
        print("Starting ETS model training...")
        model = ExponentialSmoothing(data, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=5).values  # Convert forecast to numpy array
        mse = mean_squared_error(data[-5:], forecast)
        print("ETS model training completed.")
        print("ETS model forecast generated.")
        return forecast, mse
    except Exception as e:
        print(f"ETS failed: {e}")
        return [np.nan] * 5, float('inf')


def run_linear_regression(data):
    print("\n================ Running Linear Regression Model ================\n")
    try:
        X = np.arange(len(data)).reshape(-1, 1)
        y = data.values
        model = LinearRegression()
        model.fit(X, y)
        future_X = np.arange(len(data), len(data) + 5).reshape(-1, 1)
        forecast = model.predict(future_X)
        mse = mean_squared_error(y[-5:], forecast)
        print("Linear Regression model training and forecasting completed.")
        return forecast, mse
    except Exception as e:
        print(f"Linear Regression failed: {e}")
        return [np.nan] * 5, float('inf')


def select_best_model(results):
    best_model, (best_forecast, best_mse) = min(results.items(), key=lambda x: x[1][1])
    print(f"\nBest model: {best_model} with MSE: {best_mse}\n")
    return best_model, best_forecast


def plot_forecasts(data, forecasts, best_model_name):
    print("\n================ Visualization ================\n")
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(data.index, data.values, label="Historical Data", color="black")
    
    # Create a future index for plotting forecasts
    forecast_index = pd.date_range(pd.Timestamp.today(), periods=5, freq='D')
    
    # Plot each model's forecast
    for model_name, (forecast, mse) in forecasts.items():
        if forecast is not None:
            plt.plot(forecast_index, forecast, label=f"{model_name} Forecast (MSE: {mse:.2f})")
    
    # Highlight the best model's forecast
    plt.plot(forecast_index, forecasts[best_model_name][0], color="red", linewidth=2, linestyle="--", label=f"Best Model ({best_model_name}) Forecast")
    
    # Plot formatting
    plt.xlabel("Date")
    plt.ylabel("Gold Price (USD)")
    plt.title("Gold Price Forecast by Different Models")
    plt.legend()
    
    # Save the plot as an image
    plt.savefig('forecast_plot_cleaned.png')
    print("Forecast plot saved as 'forecast_plot_cleaned.png'")


def display_forecasts_table(forecasts, best_model_name):
    print("\n================ Forecasts Table ================\n")
    
    # Create a future index with real dates starting from today in DD/MM/YYYY format
    forecast_dates = pd.date_range(pd.Timestamp.today(), periods=5, freq='D').strftime('%d/%m/%Y')
    
    forecast_data = {}
    
    for model_name, (forecast, mse) in forecasts.items():
        if forecast is not None:
            forecast_data[model_name] = forecast

    forecast_df = pd.DataFrame(forecast_data, index=forecast_dates)
    
    # Highlight the best model in the table
    forecast_df["Best Model"] = forecast_df[best_model_name]
    
    print(forecast_df)
    print("\n")


def main(data):
    # Detect and remove anomalies
    data = detect_anomalies(data)
    
    # Preprocess data
    data = preprocess_data(data)
    
    # Run different models
    forecasts = {
        "ARIMA": run_arima(data),
        "Prophet": run_prophet(data),
        "ETS": run_ets(data),
        "Linear Regression": run_linear_regression(data)
    }
    
    # Select the best model and plot the results
    best_model_name, best_forecast = select_best_model(forecasts)
    
    # Display the forecasts in a table with real dates
    display_forecasts_table(forecasts, best_model_name)
    
    # Plot the results
    plot_forecasts(data, forecasts, best_model_name)


if __name__ == "__main__":
    data = pd.read_csv("data/gold_prices.csv", parse_dates=["Date"], index_col="Date")["Close"]
    main(data)
