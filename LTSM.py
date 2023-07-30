import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf


prep_data = 'Histoy/BTC-USD/Processed/Processed data'
df = pd.read_csv(prep_data)

# Assuming you have a DataFrame named 'df' with columns: 'Open', 'High', 'Low', 'Close', '200EMA'
data = df[['open', 'high', 'low', 'close', '5EMA', '13EMA', '50EMA', '200EMA', '800EMA']].values

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Define the number of time steps (e.g., look back 5 days for predicting the next day)
time_steps = 10

# Create sequences of data for LSTM training
X, y = [], []
for i in range(len(data) - time_steps - 10):  # We use -10 to have enough data for plotting
    X.append(data[i:(i + time_steps)])
    y.append(data[i + time_steps + 10])  # Predicting 10 days in the future
X, y = np.array(X), np.array(y)

# Split the data into training and testing sets (you can choose the split ratio)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(50, input_shape=(time_steps, 9)))
model.add(Dense(9))  # Output layer with 4 neurons (Open, High, Low, Close, and 5 more EMA's)
model.compile(optimizer='adam', loss='mse')  # Use Mean Squared Error (MSE) as the loss function

# Model Training
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Model Evaluation
# Evaluate the model on the test data to see how well it performs.
loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.6f}")

# Assuming 'last_data' contains the last 'time_steps' data points in the training set
# Use it as the input for predicting the next 10 days' prices
last_data = X_train[-1].reshape(1, time_steps, 9)
predicted_prices = []

# Predict the next 10 days' prices using a loop
for i in range(10):
    prediction = model.predict(last_data)
    predicted_prices.append(prediction[0])  # Append the predicted price to the list
    last_data = np.append(last_data[:, 1:], prediction.reshape(1, 1, 9), axis=1)

# Inverse transform the predicted prices to get the actual prices
predicted_prices = scaler.inverse_transform(predicted_prices)

# Create an array with the corresponding dates for plotting
dates = pd.date_range(start=df.index[-1], periods=10)

# Plot the predicted prices using Matplotlib
plt.figure(figsize=(10, 6))
plt.plot(df['close'], label='Actual Prices')
plt.plot(predicted_prices[:, 3], label='Predicted Prices', linestyle='dashed')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Actual and Predicted Prices')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

