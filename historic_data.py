from Historic_Crypto import HistoricalData
from datetime import datetime
import yfinance as yf
import pandas as pd 
import os


# Gets Crypto Asstet
def get_user_crypto():
    user_input = input("Please enter at Crypto ticker: ")
    ticker = f'{user_input.upper()}-USD'
    time = 86400 # currently looking at daily TF
    Lookback_date = '2015-01-01-00-00'
    new = HistoricalData(ticker,time,Lookback_date).retrieve_data()
    pre_pross = pd.DataFrame(new)
    return pre_pross, ticker

def get_user_forex():
    today = datetime.today()
    
    print("\nEURUSD GBPUSD USDJPY AUDUSD USDCAD\n")
    user_input = input("\nPlease enter at Forex ticker: \n")

    ticker = f'{user_input.upper()}=x'
    # Format the date as yyyy-mm-dd
    formatted_date = today.strftime('%Y-%m-%d')
    forex_pair = "GBPUSD=X"
    # Define the start and end dates for the data you want
    start_date = "1900-01-01"
    end_date = formatted_date
    # Download the historical data
    pre_pross = yf.download(forex_pair, start=start_date, end=end_date)

    return pre_pross, ticker

def get_historical_data():
    ticker = 'BTC-USD'
    # denoted in seconds
    time = 86400 # currently looking at daily TF
    Lookback_date = '2015-01-01-00-00'

    new = HistoricalData(ticker,time,Lookback_date).retrieve_data()
    print(new)

    # Specify the directory path you want to create
    directory_path = f'Histoy/{ticker}'

    # Check if the directory already exists
    if not os.path.exists(directory_path):
        # Create the directory and any necessary parent directories
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")

    df = pd.DataFrame(new)
    df.to_csv(f"Histoy/{ticker}/{ticker}[{Lookback_date}].csv", index="False")
