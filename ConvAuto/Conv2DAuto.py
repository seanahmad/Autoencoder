# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 12:33:11 2019

@author: vhuang
"""
#%% import packages
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(42)

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LocallyConnected2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.models import clone_model

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from keras.utils import plot_model

#%%
n_epochs = 100
batch_size = 32

#%% construct convolutional neural network with
### autoencoder architecture to do 2D image compression
### load the data from keras
from tensorflow.keras.datasets import mnist

(X_train, _), (X_test, _) = mnist.load_data()

#%% normalize the data
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

print(X_train.shape)

#%% reshape the data
samples_train, height, width = X_train.shape
X_train = np.reshape(X_train, (samples_train, height, width, 1))

samples_test, height, width = X_test.shape
X_test = np.reshape(X_test, (samples_test, height, width, 1))

print(X_train.shape)

#%% construct autoencoder with CNN

### construct encoder first
input_img = Input(shape=(height, width, 1))

X = Conv2D(filters=10, kernel_size=(3,3), strides=1, 
           padding='same', activation='relu')(input_img)
X = BatchNormalization()(X)
X = MaxPooling2D(pool_size=(2,2), strides=2, padding='same')(X)

X = Conv2D(filters=5, kernel_size=(3,3), strides=1,
           padding='same', activation='relu')(X)
X = BatchNormalization()(X)

encoded = MaxPooling2D(pool_size=(2,2),  strides=2, padding='same')(X)

#%%
### construct decoder
X = Conv2D(filters=5, kernel_size=(3,3), strides=1,
           padding='same', activation='relu')(encoded)
X = BatchNormalization()(X)
X = UpSampling2D(size=(2,2))(X)

X = Conv2D(filters=10, kernel_size=(3,3), strides=1,
           padding='same', activation='relu')(X)
X = BatchNormalization()(X)
X = UpSampling2D(size=(2,2))(X)

decoded = Conv2D(filters=1, kernel_size=(3,3), strides=1,
                 padding='same', activation='relu')(X)

#%% connect encoder with decoder
Conv2DAuto = Model(inputs=[input_img], outputs=[decoded])
Conv2DAuto.summary()

#%% 
optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0.0, amsgrad=False)

checkpoint_cb = ModelCheckpoint('Conv2DAuto_model_saved.h5', 
                                save_best_only=True)

earlystopping_cb = EarlyStopping(patience=5)

#%%
Conv2DAuto.compile(loss='mse', optimizer=optimizer,
                   metrics=['mse'])

#%%
history = Conv2DAuto.fit(X_train, X_train, batch_size=batch_size,
                         epochs=n_epochs, 
                         callbacks=[checkpoint_cb, earlystopping_cb],
                         validation_split=0.2)

#%%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

#%%
X_pred = Conv2DAuto.predict(X_test)

#%%
















