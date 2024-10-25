import pandas as pd
import matplotlib.pyplot as plt

def load_and_prepare_data(file_path="data/gold_prices.csv"):
    # Step 1: Load data
    print("Loading data...")
    df = pd.read_csv(file_path)
    
    # Step 2: Convert Date column to datetime format and set as index
    print("Converting Date column to datetime format...")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Step 3: Filter relevant columns (only 'Close' price for modeling)
    if 'Close' in df.columns:
        df = df[['Close']]
    else:
        raise ValueError("The 'Close' column is missing from the data.")
    
    # Step 4: Handle missing values (if any)
    print("Checking for missing values...")
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"Found {missing_values} missing values. Filling with the last available price.")
        df.fillna(method='ffill', inplace=True)  # Forward fill missing values
    
    # Step 5: Visualize data
    print("Displaying a plot of the data...")
    plt.figure(figsize=(10, 6))
    plt.plot(df, label="Gold Close Price")
    plt.title("Historical Gold Prices")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.show()

    return df

if __name__ == "__main__":
    # Run the data preparation steps
    prepared_data = load_and_prepare_data()
