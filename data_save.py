import pandas as pd
import datetime

def save_to_csv(variable1, variable2, ticker):
    # Get today's date
    today = datetime.date.today()
    headers = ['Date', 'Prediction', 'Actual',  'Ticker']
    
    # Create a DataFrame
    data = {'Date': [today], 'Actual': [variable2], 'Prediction': [variable1], 'Ticker': ticker}
    df = pd.DataFrame(data, columns=headers)
    
    # Check if the CSV file already exists
    try:
        existing_df = pd.read_csv(f'Predictions\{ticker}_predictions.csv')
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        df.to_csv(f'Predictions\{ticker}_predictions.csv', index=False)
        return
    
    # Save the DataFrame to CSV
    df.to_csv(f'Predictions\{ticker}_predictions.csv', index=False)






