import pandas as pd

# Import the data file for adding additional features/colums for learning purposes
file = 'ETH-USD[2015-01-01-00-00].csv'
file_path = f'Histoy/ETH-USD/{file}'
df = pd.read_csv(file_path)

# Assuming you have a DataFrame named 'df' with a 'Close' price column
df['5EMA'] = df['close'].ewm(span=5, adjust=False).mean()
df['13EMA'] = df['close'].ewm(span=13, adjust=False).mean()
df['50EMA'] = df['close'].ewm(span=50, adjust=False).mean()
df['200EMA'] = df['close'].ewm(span=200, adjust=False).mean()
df['800EMA'] = df['close'].ewm(span=800, adjust=False).mean()

complete = df.to_csv(f"Histoy/BTC-USD/Processed/{file}Processed")
print(complete)

