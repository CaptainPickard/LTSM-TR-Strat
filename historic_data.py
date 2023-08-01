from Historic_Crypto import HistoricalData
import pandas as pd 
import os

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