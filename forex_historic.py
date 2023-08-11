import yfinance as yf
from datetime import datetime


forex_pairs = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"]
today = datetime.today()


forex_pair = "GBPUSD=X"
# Format the date as yyyy-mm-dd
formatted_date = today.strftime('%Y-%m-%d')
# Define the start and end dates for the data you want
start_date = "1900-01-01"
end_date = formatted_date

# Define the start and end dates for the data you want
start_date = "1900-01-01"

# Download the historical data
data = yf.download(forex_pair, start=start_date, end=end_date)

print(data)