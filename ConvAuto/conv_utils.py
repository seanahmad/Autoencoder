# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:01:36 2019

@author: vhuang
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

def drop_no_trading(data):
    dropped = []
    for i in data:
        if all(i==i[0]) == False:
            dropped.append(i)
    
    return np.array(dropped)

# load real-world time series trading data
def load_raw_data(symbol='kiwi', trading_start=510, trading_end=810):
    
    ROOT = os.path.join(r'C:\Users\vhuang\Desktop\Hazelnut', 'datasets')
    allfiles = glob.glob(os.path.join(ROOT, symbol, '*.csv'))
    
    n_samples = np.size(allfiles)
    timesteps = trading_end - trading_start
    input_dim = 1
    
    data = np.zeros((n_samples, timesteps, input_dim))
    for i, file in tqdm(enumerate(allfiles)):
        df = pd.read_csv(file, usecols=[4])
        data[i] = df.values[trading_start:trading_end]
    
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
    
    X_train = drop_no_trading(X_train)
    X_test = drop_no_trading(X_test)
    X_valid = drop_no_trading(X_valid)
    
    return X_train, X_test, X_valid
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    