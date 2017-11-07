# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 08:50:19 2017
code taken from http://parneetk.github.io/blog/cnn-cifar10/ (Written by Parneet Kaur)
@author: carmelr
"""

def weightsDiff(W1,W2):
    diff = W1
    i=0
    for l1,l2 in zip(W1,W2):
        diff[i] = W2-W1
        i=i+1
    return diff

def weightsAdd(W,W_diff):
    add = W
    i=0
    for l1,l2 in zip(W,W_diff):
        add[i] = W+W_diff
        i=i+1
    return add



###################################

#Global definitions
epochs = 8;
augmented = False

from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")


import time
import matplotlib.pyplot as plt
import numpy as np
% matplotlib inline
np.random.seed(2017) 
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator


#Load CIFAR10 Dataset
from keras.datasets import cifar10
(train_features, train_labels), (test_features, test_labels) = cifar10.load_data()
num_train, img_channels, img_rows, img_cols =  train_features.shape
num_test, _, _, _ =  test_features.shape
num_classes = len(np.unique(train_labels))


#Show Examples from Each Class
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
fig = plt.figure(figsize=(8,3))
for i in range(num_classes):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    idx = np.where(train_labels[:]==i)[0]
    features_idx = train_features[idx,::]
    img_num = np.random.randint(features_idx.shape[0])
    im = np.transpose(features_idx[img_num,::], (1, 2, 0))
    ax.set_title(class_names[i])
    plt.imshow(im)
plt.show()

#Data pre-processing
train_features = train_features.astype('float32')/255
test_features = test_features.astype('float32')/255
# convert class labels to binary class labels
train_labels = np_utils.to_categorical(train_labels, num_classes)
test_labels = np_utils.to_categorical(test_labels, num_classes)

#Function to plot model accuracy and loss
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()
    
#Funtion to compute test accuracy
def accuracy(test_x, test_y, model):
    result = model.predict(test_x)
    predicted_class = np.argmax(result, axis=1)
    true_class = np.argmax(test_y, axis=1)
    num_correct = np.sum(predicted_class == true_class) 
    accuracy = float(num_correct)/result.shape[0]
    return (accuracy * 100)



#Convolutional Neural Network for CIFAR-10 dataset
# Define the model
model = Sequential()
model.add(Conv2D(48, (3, 3), padding="same", input_shape=(3, 32, 32)))
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
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
if augmented:   
    #Use Data Augmentation
    datagen = ImageDataGenerator(zoom_range=0.2, 
                                 horizontal_flip=True)
    start = time.time()
    model_info = model.fit_generator(datagen.flow(train_features, train_labels, batch_size = 128),
                                     samples_per_epoch = train_features.shape[0], epochs = epochs, 
                                     validation_data = (test_features, test_labels), verbose=1)
    end = time.time()
else:   
    #Don't Use Data Augmentation
    start = time.time()
    model_info = model.fit(train_features, train_labels, 
                           batch_size=128, epochs=epochs, 
                           validation_data = (test_features, test_labels), 
                           verbose=1)
    end = time.time()

# Plot model history
plot_model_history(model_info)
print ('Model took %7.2f minutes to train' %((end - start)/60))

# compute test accuracy
model_accuracy = accuracy(test_features, test_labels, model)
print ("Accuracy on test data is: %5.2f" %model_accuracy)

