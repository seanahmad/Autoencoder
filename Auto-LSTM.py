#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)


timesteps = 1440
n_inputs = 1
n_epochs = 100
batch_size = 32
drop_rate = 0.8
n_samples = 6000


x = np.linspace(0, 1, timesteps)
X = np.zeros((n_samples, timesteps, 1))

for i in range(n_samples):
    z = np.random.normal()
    a = np.random.random()*10
    b = np.random.random()*5
    y = np.sin(2*np.pi*a*x) + z*10*np.sin(2*np.pi*b*x)
    X[i] = y.reshape(timesteps, 1)


for i in range(n_samples):
    plt.plot(X[i])


X_train = X[:4000]
X_test = X[4500:]
X_valid = X[4000:4500]


print(X_train.shape)
print(X_test.shape)
print(X_valid.shape)


recurrent_encoder = keras.models.Sequential([
    keras.layers.LSTM(80, return_sequences=True, input_shape=[timesteps, n_inputs]),
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.LSTM(80, return_sequences=True),
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.LSTM(30, return_sequences=True)
])


recurrent_decoder = keras.models.Sequential([
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.LSTM(80, return_sequences=True),
    keras.layers.Dropout(rate=drop_rate),
    keras.layers.LSTM(80, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(1))
])


recurrent_ae = keras.models.Sequential([
    recurrent_encoder,
    recurrent_decoder
])

recurrent_encoder.summary()

recurrent_decoder.summary()

recurrent_ae.summary()



optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                                  epsilon=None, decay=0.0, amsgrad=False)

recurrent_ae.compile(loss='mse', optimizer=optimizer,
                     metrics=['mse'])


early_stopping_cb = keras.callbacks.EarlyStopping(patience=10)
checkpoint_cb = keras.callbacks.ModelCheckpoint('recurrent_ae.h5', save_best_only=True)

# recurrent_ae = keras.models.load_model("recurrent_ae1_simulation.h5")

history = recurrent_ae.fit(X_train, X_train,
                 epochs=n_epochs, batch_size=batch_size,
                 callbacks=[checkpoint_cb, early_stopping_cb],
                 validation_data=(X_valid, X_valid))


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])


X_pred_test = recurrent_ae.predict(X_test)
X_pred_train = recurrent_ae.predict(X_train)


fig = {}
for i in range(len(X2_test)):
    if i%25 == 0:
        fig[i] = plt.figure()
        plt.plot(X2_test[i])
        plt.plot(X2_pred_test[i])
