import pandas as pd
import numpy as np

# Read the CSV file into a pandas DataFrame
data = 'Histoy/BTC-USD/BTC-USD[2015-01-01-00-00].csv'
df = pd.read_csv(data)

# Extract the 'close' prices as the target variable for prediction
target = df['close'].values

# Scale the target variable to values between 0 and 1 (optional but recommended for better model performance)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
target = scaler.fit_transform(target.reshape(-1, 1))

# Create sequences for LSTM training
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length - 10):  # -10 to predict 10 days in the future
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length + 10])  # Predicting 10 days in the future
    return np.array(X), np.array(y)

# Define the sequence length (e.g., 10 days)
sequence_length = 10

# Create sequences for LSTM training
X, y = create_sequences(target, sequence_length)

# Split the data into training and testing sets (you can choose the split ratio)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(50, input_shape=(sequence_length, 1)))
model.add(Dense(1))  # Output layer with 1 neuron for predicting the future close price
model.compile(optimizer='adam', loss='mse')  # Use Mean Squared Error (MSE) as the loss function

model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.6f}")

# Assuming 'last_data' contains the last 'sequence_length' data points in the training set
# Use it as the input for predicting the next 10 days' close prices
last_data = X_train[-1].reshape(1, sequence_length, 1)
predicted_prices = []

# Predict the next 10 days' close prices using a loop
for i in range(10):
    prediction = model.predict(last_data)
    predicted_prices.append(prediction[0][0])
    last_data = np.append(last_data[:, 1:, :], prediction.reshape(1, 1, 1), axis=1)

# Inverse transform the predicted prices to get the actual prices
predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

import matplotlib.pyplot as plt

# Create an array with the corresponding dates for plotting
dates = pd.to_datetime(df['time'].iloc[-10:]) + pd.to_timedelta(10, unit='D')

# Plot the actual close prices and the predicted close prices
plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['close'], label='Actual Close Prices')
plt.plot(dates, predicted_prices, label='Predicted Close Prices', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.title('Actual and Predicted Close Prices')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
