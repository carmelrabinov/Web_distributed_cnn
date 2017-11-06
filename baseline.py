# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 20:11:01 2017

@author: carmelr
"""

import sys
import numpy as np
import time
try: import cPickle as pickle
except: import pickle

# to aviod TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")  

import keras
from keras.models import Sequential
from keras.layers import Activation, Flatten, Dense, Dropout, Conv2D, MaxPooling2D
   

try: optimizer = sys.argv[2]
except: optimizer = 'SGD'

try: epochs = int(sys.argv[1])
except: epochs = 3

   
# Convolutional Neural Network for CIFAR-10 dataset
from keras.datasets import cifar10

input_shape=(3, 32, 32)
num_classes = 10


# Define the model
model = Sequential()
model.add(Conv2D(48, (3, 3), padding="same", input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(48, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(96, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(Conv2D(96, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(192, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(Conv2D(192, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

print('Data proccesseing...')
       
# load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# preproccessing data
num_train, img_channels, img_rows, img_cols =  x_train.shape
num_test, _, _, _ =  x_test.shape

x_train = x_train.reshape(num_train, img_channels, img_rows, img_cols)
x_test = x_test.reshape(num_test, img_channels, img_rows, img_cols)

input_shape = (img_channels, img_rows, img_cols)
x_test = x_test.astype('float32')/255
x_train = x_train.astype('float32')/255

# convert class vectors to binary class matrices
y_test = keras.utils.to_categorical(y_test, num_classes)
y_train = keras.utils.to_categorical(y_train, num_classes)
   
batch_size = 100
max_batch_num = int(num_train/batch_size)
X = []
Y = []
for i in range(int(num_train/batch_size)):
    X.append(x_train[(i * batch_size):((i+1) * batch_size - 1),:,:,:])
    Y.append(y_train[(i * batch_size):((i+1) * batch_size - 1),:])

print('Done proccesseing!')


test_lossL = []
accuracyL = []

test_loss, accuracy = model.test_on_batch(x_test,y_test)
test_lossL.append(test_loss)
accuracyL.append(accuracy)
timestamp = [0]
print('initial are test loss: {}, accuracy: {}'.format(test_loss, accuracy))


start_time = time.time()

print('Start training...')
for epoch in range(epochs):
    for batch_num in range(max_batch_num):
        model.train_on_batch(X[batch_num], Y[batch_num])
    timestamp.append(time.time() - start_time)
    test_loss, accuracy = model.test_on_batch(x_test,y_test)
    test_lossL.append(test_loss)
    accuracyL.append(accuracy)
    print(' [x] epoch {} ended with test loss: {}, accuracy: {}, time: {}'.format(epoch, test_loss, accuracy, timestamp[-1]))

print('Done training!')
   
with open('C:\\Users\\carmelr\\projectA\\baseline_results.log', 'wb') as f:
    pickle.dump([test_lossL, accuracyL, timestamp], f)
print('Dumped results')


