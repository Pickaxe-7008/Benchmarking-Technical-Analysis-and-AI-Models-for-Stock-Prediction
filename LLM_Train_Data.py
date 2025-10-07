import json
import yfinance as yf

# procure the training data
ticker_symbols = ['AAPL', 'TSLA', 'MSFT']

training_data = yf.download(
    tickers=ticker_symbols,
    period="max"
)


def Create_Data_Point(CHLOV, real_value):
    input = "Here are the last 20 days of OHLCV data:\n"

    for day, row in enumerate(CHLOV, start=1):
        input += f"Day {day} - Close: {row[0]}, High: {row[1]}, Low: {row[2]}, Open: {row[3]}, Volume: {row[4]}\n"

    return {'input': input,
            'output': str(real_value)}


def Create_Dataset(training_data):
    dataset = []
    for ticker in ticker_symbols:
        values = training_data.loc[:, (['Open', 'High', 'Low', 'Close', 'Volume'], ticker)].values

        for i in range(20, len(values)):
            CHLOV = values[i - 20:i]
            real_value = values[i][0]
            dataset.append(Create_Data_Point(CHLOV, real_value))

    with open("stock_forecasting_dataset_train_main_mk9.json", "w") as f:
        json.dump(dataset, f, indent=4)


Create_Dataset(training_data)
