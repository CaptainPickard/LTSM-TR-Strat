import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_ta as ta
from Historic_Crypto import HistoricalData
import os

# Gets Crypto Asstet
user_input = input("Please enter at crypto ticker: ")
ticker = f'{user_input.upper()}-USD'
time = 86400 # currently looking at daily TF
Lookback_date = '2015-01-01-00-00'
new = HistoricalData(ticker,time,Lookback_date).retrieve_data()
pre_pross = pd.DataFrame(new)

# Adding a "Adj close" column
pre_pross = pre_pross.assign(**{'Adj Close': pre_pross['close'].copy()})
# pre_pross['percentage_return'] = pre_pross['close'].pct_change() * 100

# Processing the data file using some TR ema indicators
pre_pross['EMA5'] = ta.ema(pre_pross.close, length=5)
pre_pross['EMA13'] = ta.ema(pre_pross.close, length=13)
pre_pross['EMA50'] = ta.ema(pre_pross.close, length=50)
pre_pross['EMA200'] = ta.ema(pre_pross.close, length=200)
pre_pross['EMA800'] = ta.ema(pre_pross.close, length=800)
pre_pross['RSI'] = ta.rsi(pre_pross.close, length=15)

# pre_pross['Target'] = pre_pross['Adj Close']-pre_pross.Open
# pre_pross['Target'] = pre_pross['Adj Close'].shift(-1)
# pre_pross['TargetClass'] = [1 if pre_pross.Target[i]>0 else 0 for i in range(len(data))]

# pre_pross['TargetNextClose'] = pre_pross['percentage_return'].shift(-1)
pre_pross['TargetNextClose'] = pre_pross['Adj Close'].shift(-1)
pre_pross.dropna(inplace=True)
pre_pross.reset_index(inplace=True)
pre_pross.drop(['volume', 'time'], axis=1, inplace=True)

# print(pre_pross.head())
print('\n**Formatting Data Complete**\n')

# Now applying the sklearn algorithm
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_set_scaled = scaler.fit_transform(pre_pross)
# print(data_set_scaled)

X = []
# choose backcandles here, originaly set to 30
backcandles = 30
# print(data_set_scaled.shape[0])
for j in range(11):
    X.append([])
    for i in range(backcandles, data_set_scaled.shape[0]):
        X[j].append(data_set_scaled[i-backcandles:i, j])

X = np.moveaxis(X, [0], [2])
X, yi = np.array(X), np.array(data_set_scaled[backcandles:,-1])
y = np.reshape(yi,(len(yi), 1))


print('\n**Splitting data into Testing and Training sets**\n')

splitlimit = int(len(X)*0.8)
X_train, X_test = X[:splitlimit], X[:splitlimit]
y_train, y_test = y[:splitlimit], y[:splitlimit]


print('\n**Fitting the Model**\n')

from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Input, Activation

import tensorflow as tf
from keras import optimizers
from keras.models import Model
import numpy as np

X_train_normalized = tf.keras.utils.normalize(X_train, axis=1)
X_test_normalized = tf.keras.utils.normalize(X_test, axis=1)

lstm_input = Input(shape=(backcandles, 11), name='LSTM_Input')
inputs = LSTM(150, name='first_layer')(lstm_input)
inputs = Dense(1, name='dense_layer')(inputs)
output = Activation('linear', name='output')(inputs)
model = Model(inputs=lstm_input, outputs=output)
adam = optimizers.Adam()
model.compile(optimizer=adam, loss='mse')
model.fit(x=
X_test_normalized, y=y_train, batch_size=15, epochs=50, shuffle=True, validation_split = 0.1)

y_pred = model.predict(X_test)

scaling_factors = np.max(y_train, axis=0)

# Inverse transform the scaled predictions 'y_pred' to the original scale
y_pred_original_scale = y_pred * scaling_factors

# Plotting 4 different timeframes of the output data
# Last 500 Rows of Price
x_close_500 = pre_pross['close'][-500:]
x_ema_500 = pre_pross['EMA50'][-500:]
x_ema_200 = pre_pross['EMA200'][-500:]
x_ema_800 = pre_pross['EMA800'][-500:]
# time_500 = pre_pross['time'][-500:]

# Last 500 Rows of Dataset
y_test_500 = y_test[-500:]
y_pred_500 = y_pred_original_scale[-500:]

# Last 50 Rows of Price
x_close_50 = pre_pross['close'][-50:]
x_ema_50 = pre_pross['EMA50'][-50:]
x_ema_5 = pre_pross['EMA5'][-50:]
x_ema_13 = pre_pross['EMA13'][-50:]
# time_50 = pre_pross['time'][-50:]

# Last 50 Rows of Dataset
y_test_50 = y_test[-50:]
y_pred_50 = y_pred_original_scale[-50:]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# PRICE LONG
axes[0, 0].plot(x_close_500, color = 'black', label = 'Close Price')
axes[0, 0].plot(x_ema_500, color = 'green', label = '50 EMA')
axes[0, 0].plot(x_ema_200, color = 'red', label = '200 EMA')
axes[0, 0].plot(x_ema_800, color = 'blue', label = '800 EMA')
axes[0, 0].legend()
axes[0, 0].set_title('Last 500 Rows of Price')
axes[0, 0].grid(True)

# PREDICTION LONG
axes[0, 1].plot(y_test_500, color = 'black', label = 'Test')
axes[0, 1].plot(y_pred_500, color = 'red', label = 'Pred')
axes[0, 1].legend()
axes[0, 1].set_title('Last 500 Rows of Dataset')
axes[0, 1].grid(True)

# PRICE SHORT
axes[1, 0].plot(x_close_50, color = 'black', label = 'Close Price')
axes[1, 0].plot(x_ema_5, color = 'green', label = '5 EMA')
axes[1, 0].plot(x_ema_13, color = 'red', label = '13 EMA')
axes[1, 0].plot(x_ema_50, color = 'blue', label = '50 EMA')
axes[1, 0].legend()
axes[1, 0].set_title('Last 50 Rows of Price')
axes[1, 0].grid(True)

# PREDICTION SHORT
axes[1, 1].plot(y_test_50, color = 'black', label = 'Test')
axes[1, 1].plot(y_pred_50, color = 'red', label = 'Pred')
axes[1, 1].legend()
axes[1, 1].set_title('Last 50 Rows of Dataset')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()



