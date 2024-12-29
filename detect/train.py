import numpy as np
import pandas as pd

from keras.layers import LSTM, Dense,Dropout
from keras.models import Sequential
import tensorflow as tf
ADL_train = pd.read_csv("./dataset/HFD/kinetics/ADL/train/ADL.csv")
Fall_train = pd.read_csv("./dataset/HFD/kinetics/Fall/train/Fall.csv")
ADL_test = pd.read_csv("./dataset/HFD/kinetics/ADL/test/ADL.csv")
Fall_test = pd.read_csv("./dataset/HFD/kinetics/Fall/test/Fall.csv")

X_train = []
y_train = []

X_test = []
y_test = []
no_of_timesteps = 10
for i in range(no_of_timesteps, ADL_train.iloc[:,1:].values.shape[0]):
    X_train.append(ADL_train.iloc[:,1:].values[i-no_of_timesteps:i,:])
    y_train.append(0)

for i in range(no_of_timesteps, Fall_train.iloc[:,1:].values.shape[0]):
    X_train.append(Fall_train.iloc[:,1:].values[i-no_of_timesteps:i,:])
    y_train.append(1)


for i in range(no_of_timesteps, ADL_test.iloc[:,1:].values.shape[0]):
    X_test.append(ADL_test.iloc[:,1:].values[i-no_of_timesteps:i,:])
    y_test.append(0)
for i in range(no_of_timesteps, Fall_test.iloc[:,1:].values.shape[0]):
    X_test.append(Fall_test.iloc[:,1:].values[i-no_of_timesteps:i,:])
    y_test.append(1)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

with tf.device('/GPU:0'): 
    model  = Sequential()
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1] + X_test.shape[1],X_train.shape[2] + X_test.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    model.add(Dense(units = 1, activation="sigmoid"))
    model.compile(optimizer="adam", metrics = ['accuracy'], loss = "binary_crossentropy")

    model.fit(X_train, y_train, epochs=16, batch_size=32,validation_data=(X_test, y_test))
    model.save("model.h5")