import pandas as pd

# Import the data file for adding additional features/colums for learning purposes
df = pd.DataFrame('Histoy/BTC-USD/BTC-USD[2015-01-01-00-00].csv')

# Assuming you have a DataFrame named 'df' with a 'Close' price column
df['5EMA'] = df['Close'].ewm(span=5, adjust=False).mean()
df['13EMA'] = df['Close'].ewm(span=13, adjust=False).mean()
df['50EMA'] = df['Close'].ewm(span=50, adjust=False).mean()
df['200EMA'] = df['Close'].ewm(span=200, adjust=False).mean()
df['800EMA'] = df['Close'].ewm(span=800, adjust=False).mean()


