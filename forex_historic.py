import yfinance as yf

forex_pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"]

forex_pair = "GBPUSD=X"
# Define the start and end dates for the data you want
start_date = "1900-01-01"
end_date = "2023-08-01"

# Download the historical data
data = yf.download(forex_pair, start=start_date, end=end_date)

print(data)