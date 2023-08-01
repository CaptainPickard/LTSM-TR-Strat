import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_ta as ta

file = ('Histoy\ETH-USD\ETH-USD[2015-01-01-00-00].csv')
pre_pross = pd.read_csv(file)

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

# getting the closing price of the 5 days

# pre_pross['TargetNextClose'] = pre_pross['percentage_return'].shift(-1)
pre_pross['TargetNextClose'] = pre_pross['Adj Close'].shift(-1)
pre_pross.dropna(inplace=True)
pre_pross.reset_index(inplace=True)
pre_pross.drop(['volume', 'close', 'time'], axis=1, inplace=True)

# print(pre_pross.head())
print('\n**Formatting Data Complete**\n')

# Now applying the sklearn algorithm
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_set_scaled = scaler.fit_transform(pre_pross)
# print(data_set_scaled)

X = []
# choose backcandles here, originaly set to 30
backcandles = 50
# print(data_set_scaled.shape[0])
for j in range(10):
    X.append([])
    for i in range(backcandles, data_set_scaled.shape[0]):
        X[j].append(data_set_scaled[i-backcandles:i, j])

X = np.moveaxis(X, [0], [2])
X, yi = np.array(X), np.array(data_set_scaled[backcandles:,-1])
y = np.reshape(yi,(len(yi), 1))

# Verifying the shape of the data set to feed into the model
# print(X.shape)
# print(y.shape)

print('\n**Splitting data into Testing and Training sets**\n')

splitlimit = int(len(X)*0.8)
X_train, X_test = X[:splitlimit], X[:splitlimit]
y_train, y_test = y[:splitlimit], y[:splitlimit]


print('\n**Fitting the Model**\n')

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Input, Activation, concatenate

import tensorflow as tf
import keras 
from keras import optimizers
from keras.callbacks import History
from keras.models import Model
import numpy as np

lstm_input = Input(shape=(backcandles, 10), name='LSTM_Input')
inputs = LSTM(150, name='first_layer')(lstm_input)
inputs = Dense(1, name='dense_layer')(inputs)
output = Activation('linear', name='output')(inputs)
model = Model(inputs=lstm_input, outputs=output)
adam = optimizers.Adam()
model.compile(optimizer=adam, loss='mse')
model.fit(x=X_train, y=y_train, batch_size=15, epochs=50, shuffle=True, validation_split = 0.1)

y_pred = model.predict(X_test)

# Plotting 4 different timeframes of the output data

y_test_1000 = y_test[-1000:]
y_pred_1000 = y_pred[-1000:]

y_test_500 = y_test[-500:]
y_pred_500 = y_pred[-500:]

y_test_100 = y_test[-100:]
y_pred_100 = y_pred[-100:]

y_test_50 = y_test[-50:]
y_pred_50 = y_pred[-50:]

fig, axes = plt.subplots(2, 2, figsize=(15, 8))

axes[0, 0].plot(y_test_1000, color = 'black', label = 'Test')
axes[0, 0].plot(y_pred_1000, color = 'green', label = 'Pred')
axes[0, 0].legend()
axes[0, 0].set_title('Last 1000 Rows of Dataset')
axes[0, 0].grid(True)

axes[0, 1].plot(y_test_500, color = 'black', label = 'Test')
axes[0, 1].plot(y_pred_500, color = 'orange', label = 'Pred')
axes[0, 1].legend()
axes[0, 1].set_title('Last 500 Rows of Dataset')
axes[0, 1].grid(True)

axes[1, 0].plot(y_test_100, color = 'black', label = 'Test')
axes[1, 0].plot(y_pred_100, color = 'blue', label = 'Pred')
axes[1, 0].legend()
axes[1, 0].set_title('Last 100 Rows of Dataset')
axes[1, 0].grid(True)

# axes[1, 1].plot(pre_pross['close'], color = 'black', label = 'Actual')
axes[1, 1].plot(y_test_50, color = 'black', label = 'Test')
axes[1, 1].plot(y_pred_50, color = 'red', label = 'Pred')
axes[1, 1].legend()
axes[1, 1].set_title('Last 50 Rows of Dataset')
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()



