import yfinance as yf
from datetime import datetime


def get_user_forex():
    today = datetime.today()
    
    print("\n EURUSD GBPUSD USDJPY AUDUSD USDCAD\n")
    user_input = input("\n Please enter at Forex ticker: ")

    ticker = f'{user_input.upper()}=x'
    formatted_date = today.strftime('%Y-%m-%d')
    forex_pair = ticker
    
    user_input_year = input("\n Please enter a starting year (1900 for all time): ")
    start_date = f"{user_input_year}-01-01"
    end_date = formatted_date
    pre_pross = yf.download(forex_pair, start=start_date, end=end_date)
    print(pre_pross)
