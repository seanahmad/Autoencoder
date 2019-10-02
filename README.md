# Autoencoder & Dimension Reduction

The goal of this project is to achieve dimension reduction. Autoencoder is usually used to do dimension reduction and feature extraction. 
  
The data I used is the intraday data with 1440 minutes(rows). Each row in the time series data is a price. To eliminate the noise overnight, only the day session from market open to close is used. The symbol I used in the demo is GC(gold futures). It has 300 minutes during the daytime trading. The objective is to reduce this 300 dimension series into a lower dimension. The target I achieved is a (5, 8) matrix. Flattened, it is a 40 dimension series.

The repository include:  
An autoencoder architecture with convolutional neural network to do dimension reduction.  
<a href="https://gqhuang.com/auto-4/">Conv-Auto</a> Here is included the details about how it works.  

An autoencoder architecture with recurrent neural network to do feature extraction.  
<a href="https://gqhuang.com/auto-3/">Auto-LSTM</a> Here is included the details about how it works.  

All the architectures are built with __*TensorFlow*__ and __*Keras*__.

## Convolutional-Autoencoder

This architecture is a combination of convolutional layers and autoencoder. Keras has a type of convolutional layer called `Conv1D` used to deal with sequence data. A `Conv1D` layer has different filters and it applies these different filters upon the sequence data, with the filters sliding along the sequence data. It works in a similar way with the common `Conv2D`. 
  
Firstly, the encoder with 4 `Conv1D` layers, each followed by a `Maxpooling1D` layer, compresses the 300 dimension sequence into an abstract array of dimension `(5,8)`. Secondly, the decoder with another 4 symmetrical `Conv1D` layers, each followed by a `Upsampling1D` layer, reconstruct the original input from the compressed array.

More technical details about how to build up the architecture refer to the latest Jupyter Notebook file named `Conv1DAuto_V4_daySession.ipynb` above.

### How the reproduce looks like?
Below are some comparision between the original input and the reproduce output

![](https://raw.githubusercontent.com/VictorXXXXX/Autoencoder/master/images/result1.png)
![](https://raw.githubusercontent.com/VictorXXXXX/Autoencoder/master/images/result2.png)
![](https://raw.githubusercontent.com/VictorXXXXX/Autoencoder/master/images/result3.png)
![](https://raw.githubusercontent.com/VictorXXXXX/Autoencoder/master/images/result4.png)
![](https://raw.githubusercontent.com/VictorXXXXX/Autoencoder/master/images/result5.png)
![](https://raw.githubusercontent.com/VictorXXXXX/Autoencoder/master/images/result6.png)

### How the codings looks like?
After training the model to finish the reconstruction task, I cut the decoder and used the encoder to fit the test data into lower dimension codings. Below is the correlation plot of the codings.

![](https://github.com/VictorXXXXX/Autoencoder/blob/master/images/corr.png)

https://raw.githubusercontent.com/VictorXXXXX/Autoencoder/master/images/corr.png
## LSTM-Autoencoder

This architecture is the first one I try to use to do dimension reduction on sequence data. The main reason I used LSTM in an autoencoder is that LSTM is capable of capturing time dependent features and patterns. But I failed to finish the compression goal, because as I learned more about LSTM, I found that when combined with autoencoder, it could not compress the sequence dimension. It only deals with the features size. In other words, everytime you pass a timestep into LSTM, you got a timestep out. The dimension of timestep does not change at all. But it may still be useful. It may be able to be used to do feature extraction. Say, you now have a sequence of data with 5 features. Passed into the encoder, the data is expanded into a larger dimension, with 10 features. 
