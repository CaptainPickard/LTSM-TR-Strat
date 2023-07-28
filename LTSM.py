import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras


df = pd.DataFrame('Prepared dataset')

# Assuming you have a DataFrame named 'df' with columns: 'Open', 'High', 'Low', 'Close', '200EMA'
data = df[['Open', 'High', 'Low', 'Close', '5EMA', '13EMA', '50EMA', '200EMA', '800EMA']].values

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Define the number of time steps (e.g., look back 5 days for predicting the next day)
time_steps = 5

# Create sequences of data for LSTM training
X, y = [], []
for i in range(len(data) - time_steps - 1):
    X.append(data[i:(i + time_steps)])
    y.append(data[i + time_steps])
X, y = np.array(X), np.array(y)

# Split the data into training and testing sets (you can choose the split ratio)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]


from keras.models import Sequential
from keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(50, input_shape=(time_steps, 4)))
model.add(Dense(4))  # Output layer with 4 neurons (Open, High, Low, Close)
model.compile(optimizer='adam', loss='mse')  # Use Mean Squared Error (MSE) as the loss function

# Model Training
model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

# Model Evaluation
# Evaluate the model on the test data to see how well it performs.
loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.6f}")


# Assuming 'last_data' contains the last 'time_steps' data points in the training set
# Use it as the input for predicting the next day's price
last_data = X_train[-1].reshape(1, time_steps, 9)  # Nine features: Open, High, Low, Close, 5EMA - 800EMA
predicted_price = model.predict(last_data)

# Inverse transform the predicted price to get the actual price
predicted_price = scaler.inverse_transform(predicted_price)
print("Predicted Price for the Next Day:")
print(predicted_price)


