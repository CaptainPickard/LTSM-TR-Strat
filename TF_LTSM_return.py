import numpy as np
import pandas as pd
import pandas_ta as ta
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

print('--------- Step 2: Prepare the data')

file = ('Histoy\BTC-USD\BTC-USD[2015-01-01-00-00].csv')
df = pd.read_csv(file)
# Calculate the price returns as the target variable
df['EMA50'] = ta.ema(df['close'], length=50)
# Extract the features and target (returns)
df.dropna(inplace=True)
df.reset_index(inplace=True)
df.drop(['index', 'time', 'volume'], axis=1, inplace=True)
df['percentage_return'] = df['close'].pct_change() * 100
# Separate input features and target variable
X = df[['low', 'high', 'open', 'close', 'EMA50', 'percentage_return']].values
y = df['percentage_return'].values.reshape(-1, 1)
df.dropna(inplace=True)

print('--------- Step 3: Standardize the input features')

# Create a StandardScaler object
scaler = StandardScaler()
# Standardize the input features
X_scaled = scaler.fit_transform(X)

print('--------- Step 4: Convert the target variable to percentage return')

# Assuming the 'percentage_return' values are decimal values, not percentages
# Convert the target variable to percentage return by multiplying by 100
y_percentage_return = y * 100

print('--------- Step 5: Split the data into training and testing sets')

# Split the data into training and testing sets (you can choose the split ratio)
train_size = int(0.8 * len(X_scaled))
X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
y_train, y_test = y_percentage_return[:train_size], y_percentage_return[train_size:]

print('--------- Step 6: Build the TensorFlow model')

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')  # Use Mean Squared Error (MSE) as the loss function

print('--------- Step 7: Train the model')

model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)


print('--------- Step 8: Make predictions and convert back to percentage returns')

# Predict the percentage returns on the test data
y_pred = model.predict(X_test)
# Convert the predicted percentage returns back to raw decimal values
y_pred_decimal = y_pred / 100


print('--------- Step 9: Inverse transform the scaled output')

# Inverse transform the predicted output to the original scale
y_pred_original_scale = scaler.inverse_transform(y_pred_decimal)
# Assuming 'last_data' contains the last 5 data points in the training set
# Use it as the input for predicting the next day's percentage return
last_data = X_train[-1].reshape(1, -1)
next_day_percentage_return = model.predict(last_data)
# Convert the predicted percentage return back to raw decimal value
next_day_percentage_return_decimal = next_day_percentage_return / 100
# Inverse transform the scaled percentage return to the original scale
next_day_percentage_return_original_scale = scaler.inverse_transform(next_day_percentage_return_decimal)
print(next_day_percentage_return_original_scale)

