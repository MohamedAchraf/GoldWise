import requests
import pandas as pd
import yfinance as yf
import os

# Define API Keys and file path
ALPHA_API_KEY = "8E08KRXT8U5EBW3Z"
OUTPUT_FILE = "data/gold_prices.csv"

def fetch_alpha_vantage(api_key, symbol="XAUUSD"):
    """
    Fetch historical gold prices from Alpha Vantage.
    """
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "full",
        "apikey": api_key
    }
    response = requests.get(url, params=params)

    if response.status_code == 200:
        print("Alpha Vantage data fetched successfully.")
        data = response.json().get("Time Series (Daily)", {})
        df_alpha = pd.DataFrame.from_dict(data, orient="index")
        df_alpha.columns = ["Open", "High", "Low", "Close", "Volume"]
        df_alpha.index = pd.to_datetime(df_alpha.index).tz_localize(None)  # Remove timezone awareness
        df_alpha = df_alpha.apply(pd.to_numeric)
        return df_alpha[["Open", "High", "Low", "Close"]]
    else:
        print("Failed to fetch Alpha Vantage data.")
        return None

def fetch_yahoo_finance(symbol="GC=F"):
    """
    Fetch historical gold prices from Yahoo Finance.
    """
    print("Fetching data from Yahoo Finance...")
    df_yahoo = yf.download(symbol, start="2000-01-01", end="2024-12-31")
    if not df_yahoo.empty:
        df_yahoo = df_yahoo[["Open", "High", "Low", "Close"]]
        df_yahoo.index = df_yahoo.index.tz_localize(None)  # Ensure timezone-naive timestamps
        return df_yahoo
    else:
        print("Failed to fetch Yahoo Finance data.")
        return None

def fetch_and_save_data(api_key, output_file=OUTPUT_FILE):
    """
    Fetch gold data from both Alpha Vantage and Yahoo Finance, merge, and save it.
    """
    alpha_vantage_data = fetch_alpha_vantage(api_key)
    yahoo_finance_data = fetch_yahoo_finance()

    if alpha_vantage_data is not None and not alpha_vantage_data.empty:
        print("Combining data from both sources...")
        combined_data = pd.concat([alpha_vantage_data, yahoo_finance_data]).drop_duplicates()
        combined_data.sort_index(inplace=True)

        print("Saving combined data to CSV...")
        combined_data.to_csv(output_file, index_label="Date")
        print(f"Data saved to {output_file}")
    else:
        print("Error: No data retrieved or data is empty.")

if __name__ == "__main__":
    fetch_and_save_data(ALPHA_API_KEY)
