import pandas as pd
import pandas_ta as ta

print('---- Step 1: Data Preparation ----')

file = ('Histoy\BTC-USD\BTC-USD[2015-01-01-00-00].csv')
df = pd.read_csv(file)
# Calculate the price returns as the target variable
df['EMA50'] = ta.ema(df['close'], length=50)
df['returns'] = df['close'].pct_change().fillna(0)
returns = df['close'].pct_change().values[1:]

print('---- Step 2: Standardization ----')

from sklearn.preprocessing import StandardScaler
# Extract the features for standardization (all except 'time' and 'returns')
features = df[['open', 'high', 'low', 'close', 'EMA50']].values
# Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

print('---- Step 3: Data Splitting ----')

import numpy as np
# Define the number of time steps (e.g., look back 10 days for predicting the next day)
time_steps = 10
# Create sequences of data for LSTM training
X, y = [], []
for i in range(len(scaled_features) - time_steps - 1):
    X.append(scaled_features[i:(i + time_steps)])
    y.append(df['returns'].values[i + time_steps])
X, y = np.array(X), np.array(y)

# Split the data into training and testing sets (you can choose the split ratio)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


print('---- Step 4: Build and Train the LSTM Model ----')

import torch
import torch.nn as nn
import torch.optim as optim

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Set hyperparameters
input_size = X_train.shape[2]
hidden_size = 50
output_size = 1

# Initialize the model
model = LSTMModel(input_size, hidden_size, output_size)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs.squeeze(), y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}")


print('---- Step 5: Model Evaluation ----')

# Convert the test data to PyTorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Evaluate the model
with torch.no_grad():
    model.eval()
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs.squeeze(), y_test_tensor)
    print(f"Test Loss: {test_loss.item():.6f}")


print('---- Step 6: Model Eval and Price Prediction ----')

# Assuming 'last_data' contains the last 'time_steps' data points in the training set
# Use it as the input for predicting the next day's return value
last_data = X_train[-1].reshape(1, time_steps, input_size)

# Convert 'last_data' to a PyTorch tensor
last_data_tensor = torch.tensor(last_data, dtype=torch.float32)

# Evaluate the model and make predictions for the next day's return value
with torch.no_grad():
    model.eval()
    next_day_return = model(last_data_tensor).item()

# Inverse transform the predicted return to get the actual return value
next_day_return = scaler.inverse_transform([[next_day_return]])[1][5]

print("Predicted Return for the Next Day:")
print(next_day_return)