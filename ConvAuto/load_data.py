# -*- coding: utf-8 -*-
"""
Spyder Editor

author: victor
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
np.random.seed(42)
from tqdm import tqdm

def preprocess(data):
    for i in range(data.shape[0]):
        scaler = StandardScaler()
        data[i] = scaler.fit_transform(data[i])
    return data

# load real-world time series trading data
def load_real_data(symbol):
    
    ROOT = os.path.join(r'C:\Users\vhuang\Desktop\Hazelnut', 'datasets')
    allfiles = glob.glob(os.path.join(ROOT, symbol, '*.csv'))
    
    n_samples = np.size(allfiles)
    timesteps = 1440
    input_dim = 1
    
    data = np.zeros((n_samples, timesteps, input_dim))
    for i, file in tqdm(enumerate(allfiles)):
        df = pd.read_csv(file, usecols=[4])
        if df.values.shape[0] != timesteps:
            print(file)
            print(i + ' missing rows')
        else:
            data[i] = df.values
    
    # training set is 0.8 of the total
    test_split = int(data.shape[0]*0.8)
    X_train_full = data[:test_split]
    X_test = data[test_split:]
    # validation set is 0.1 of the training
    valid_split = int(test_split*0.9)
    X_train = X_train_full[:valid_split]
    X_valid = X_train_full[valid_split:]

    X_train = preprocess(X_train)
    X_test = preprocess(X_test)
    X_valid = preprocess(X_valid)
    
    return X_train, X_test, X_valid


# generate simulation sequence using trigonometric functions
def load_simulation_data():
    
    n_samples = 6000
    timesteps = 1440
    
    x = np.linspace(0, 1, timesteps)
    X = np.zeros((n_samples, timesteps, 1))
    
    for i in tqdm(range(n_samples)):
        z = np.random.normal()
        a = np.random.random()*10
        b = np.random.random()*5
        y = np.sin(2*np.pi*a*x) + z*10*np.sin(2*np.pi*b*x)
        X[i] = y.reshape(timesteps, 1)

    X_train = X[:4000]
    X_test = X[4500:]
    X_valid = X[4000:4500]

    X_train = preprocess(X_train)
    X_test = preprocess(X_test)
    X_valid = preprocess(X_valid)
    
    return X_train, X_test, X_valid

















