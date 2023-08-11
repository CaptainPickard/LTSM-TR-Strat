import yfinance as yf
from datetime import datetime


def get_user_forex():
    today = datetime.today()
    
    print("\nEURUSD GBPUSD USDJPY AUDUSD USDCAD\n")
    user_input = input("\nPlease enter at Forex ticker: \n")

    ticker = f'{user_input.upper()}=x'
    # Format the date as yyyy-mm-dd
    formatted_date = today.strftime('%Y-%m-%d')
    forex_pair = ticker
    # Define the start and end dates for the data you want
    start_date = "1900-01-01"
    end_date = formatted_date
    # Download the historical data
    pre_pross = yf.download(forex_pair, start=start_date, end=end_date)
    print(pre_pross)
    # return pre_pross, ticker

get_user_forex()