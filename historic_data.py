from Historic_Crypto import HistoricalData
from datetime import datetime
import yfinance as yf
import pandas as pd 
import os


# Gets Crypto Asstet
def get_user_crypto():
    user_input = input("\nPlease enter at Crypto ticker: ")
    ticker = f'{user_input.upper()}-USD'
    time = 86400 # currently looking at daily TF
    Lookback_date = '2015-01-01-00-00'
    new = HistoricalData(ticker,time,Lookback_date).retrieve_data()
    pre_pross = pd.DataFrame(new)
    return pre_pross, ticker


def get_user_forex():
    today = datetime.today()
    print("\nEURUSD GBPUSD USDJPY AUDUSD USDCAD\n")
    user_input = input("\nPlease enter at Forex ticker: ")
    ticker = f'{user_input.upper()}=x'
    formatted_date = today.strftime('%Y-%m-%d')
    forex_pair = ticker
    start_date = "1900-01-01"
    end_date = formatted_date
    pre_pross = yf.download(forex_pair, start=start_date, end=end_date)
    return pre_pross, ticker

