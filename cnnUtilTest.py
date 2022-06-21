import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten
from keras import backend as k
import matplotlib.pyplot as plt

n_frames, n_bins, n_hidden, dropout = 270, 184, 64, 0
input_shape = (n_bins, n_hidden, 1)

print(input_shape)#(28, 28, 1) on the mnist cnn
#(184, 64, 1) on cnnUtilTest

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),activation='relu', data_format = 'channels_last',input_shape=input_shape))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

'''
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3),activation='relu', data_format = 'channels_last',input_shape=input_shape))
model.add(Conv2D(64, (3,3), activation='relu'))

#model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(dropout))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(dropout))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
#'''
