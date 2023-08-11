import matplotlib.pyplot as plt
from keras.layers import Input, Activation, Dense, LSTM
import tensorflow as tf
from keras import optimizers
from keras.models import Model
import numpy as np
from sklearn.preprocessing import StandardScaler

def mlm_model(pre_pross):
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

    X_train_normalized = tf.keras.utils.normalize(X_train, axis=1)
    X_test_normalized = tf.keras.utils.normalize(X_test, axis=1)

    lstm_input = Input(shape=(backcandles, 11), name='LSTM_Input')
    inputs = LSTM(150, name='first_layer')(lstm_input)
    inputs = Dense(1, name='dense_layer')(inputs)
    output = Activation('linear', name='output')(inputs)
    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam()
    model.compile(optimizer=adam, loss='mse')
    model.fit(x=X_train_normalized, 
              y=y_train, 
              batch_size=15, 
              epochs=50, 
              shuffle=True, 
              validation_split = 0.1)

    y_pred = model.predict(X_test_normalized)
    train_x = model.predict(X_train_normalized)

    scaling_factors_pred = np.max(y_train, axis=0)
    scaling_factors_train = np.max(y_test, axis=0)

    # Inverse transform the scaled predictions 'y_pred' to the original scale
    y_pred_original_scale = y_pred * scaling_factors_pred
    # y_train_original_scale = train_x * scaling_factors_train
    
    # for middle return value y_train was used previously
    return pre_pross, y_train, y_pred_original_scale
