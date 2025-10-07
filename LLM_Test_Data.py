import json
import yfinance as yf
import numpy as np
import pandas as pd

# Download training data
ticker_symbols = ['GOOG']
training_data = yf.download(ticker_symbols, period="max")

def Create_Data_Point(CHLOV, real_value):
    input_str = "Here are the last 20 days of OHLCV data:\n"
    for day, row in enumerate(CHLOV, start=1):
        # Ensure numbers are properly formatted
        close, high, low, open_, volume = row
        input_str += (
            f"Day {day} - Close: {close:.2f}, High: {high:.2f}, "
            f"Low: {low:.2f}, Open: {open_:.2f}, Volume: {int(volume)}\n"
        )

    return {
        "input": input_str.strip(),
        "output": round(float(real_value), 4)  # numeric output
    }

def Create_Dataset(training_data):
    dataset = []

    # If only one ticker, columns are not MultiIndex
    if isinstance(training_data.columns, pd.MultiIndex):
        for ticker in ticker_symbols:
            values = training_data.loc[:, (['Close', 'High', 'Low', 'Open', 'Volume'], ticker)].values
            for i in range(20, len(values)):
                CHLOV = values[i - 20:i]
                real_value = values[i][0]  # close price
                dataset.append(Create_Data_Point(CHLOV, real_value))
    else:
        # Flat columns (single ticker)
        values = training_data[['Close', 'High', 'Low', 'Open', 'Volume']].values
        for i in range(20, len(values)):
            CHLOV = values[i - 20:i]
            real_value = values[i][0]  # close price
            dataset.append(Create_Data_Point(CHLOV, real_value))

    # Save clean JSON
    with open("stock_forecasting_dataset_test_main_mk2.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)

    print(f"✅ Dataset saved successfully with {len(dataset)} entries.")

Create_Dataset(training_data)
