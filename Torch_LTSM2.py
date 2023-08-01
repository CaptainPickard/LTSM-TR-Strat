import pandas as pd
import pandas_ta as ta

print('---- Step 1: Data Preparation ----')

file = ('Histoy\BTC-USD\BTC-USD[2015-01-01-00-00].csv')
df = pd.read_csv(file)
# Calculate the price returns as the target variable
df['EMA50'] = ta.ema(df['close'], length=50)
# Extract the features and target (returns)
df.dropna(inplace=True)
df.reset_index(inplace=True)
df.drop(['index', 'time', 'volume'], axis=1, inplace=True)
features = df[['low', 'high', 'open', 'close', 'EMA50']].values
returns = df['close'].pct_change().values[1:]  # Calculate returns as percentage change from the previous day's close
# print(features, returns)
# print(df.head())


print('---- Step 2: Standardize the Data ----')

from sklearn.preprocessing import StandardScaler
# Standardize the features using Z-score scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)



print('---- Step 3: Prepare Sequences for LSTM Training ----')

import numpy as np

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length - 1):  # -1 to predict returns for the next day
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length + 1])  # Predicting returns for the next day
    X, y = np.array(X), np.array(y)
    return X.reshape(-1, seq_length, X.shape[1]), y

# Define the sequence length (e.g., 10 days)
sequence_length = 10
while 2929 % sequence_length != 0:
    sequence_length -= 1
# Create sequences for LSTM training
X, y = create_sequences(scaled_features, sequence_length)



print('---- Step 4: Create and Train the LSTM Model ----')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Create a DataLoader for batch training
train_data = TensorDataset(X, y)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)

# Define the LSTM model
# for batch_X, batch_y in train_loader:
#     print("Batch X shape:", batch_X.shape)
#     print("Batch y shape:", batch_y.shape)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

# Initialize the model
# input_size = features.shape[1]
input_size = scaled_features.shape[1]
hidden_size = 50
output_size = 1  # Predicting returns
model = LSTMModel(input_size, hidden_size, output_size)
# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.view(-1), batch_y)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")


print('---- Step 5: Predict the Returns ----')

# Predict the returns for the next day
model.eval()
with torch.no_grad():
    last_data = X[-1].reshape(1, sequence_length, input_size)
    predicted_returns = []
    for i in range(10):
        prediction = model(last_data)
        predicted_returns.append(prediction.item())
        last_data = torch.cat([last_data[:, 1:, :], prediction.reshape(1, 1, input_size)], dim=1)

# Convert predicted returns back to price predictions using scaler.inverse_transform()
predicted_returns = np.array(predicted_returns).reshape(-1, 1)
predicted_prices = scaler.inverse_transform(predicted_returns)

# Calculate the price predictions for 10 days in the future
last_close_price = df['close'].iloc[-1]
price_predictions = last_close_price * (1 + predicted_returns.cumsum())

# print(last_close_price, price_predictions)

