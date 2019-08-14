# Autoencoder


The goal of this project is to achieve dimension reduction. Autoencoder is usually used to do dimension reduction and feature extraction.  
The data I used is the intraday data with 1440 minutes(rows). Each row in the time series data is a price. The objective is to reduce this 1440 dimension series into a lower dimension. The target I achieved is a (5, 8) matrix. Flattened, it is a 40 dimension series.

The repository include:  
An autoencoder architecture with convolutional neural network to do dimension reduction <a href="https://gqhuang.com/auto-4/">Conv-Auto</a>  
An autoencoder architecture with recurrent neural network to do feature extraction <a href="https://gqhuang.com/auto-3/">Auto-LSTM</a>  

All the architectures are built with Tensorflow and Keras.

## Convolutional-Autoencoder

This architecture is a combination of convolutional layers and autoencoder. Keras has a type of convolutional layer called `Conv1D` used to deal with sequence data. A Conv1D layer has different filters and it applies these different filters upon the sequence data, with the filters sliding along the sequence data. It works in a similar way with the common `Conv2D`. 
  
Firstly, the encoder with a bunch of `Conv1D` layers and `MaxPooling1D` layers compresses the 1440 dimension sequence into an abstract array of dimension `(5,8)`. Secondly, the decoder with another symmetrical bunch of `Upsampling1D` layers and `Conv1D` layers to do reconstruction.

### Original Input and Reproduce output
Below are some comparision between the original input and the reproduce output


