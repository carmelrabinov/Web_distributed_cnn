# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 20:11:01 2017

@author: carmelr
"""

import sys
import numpy as np
import time
import matplotlib.pyplot as plt
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from smtplib import SMTP
import smtplib

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
from keras.optimizers import SGD
   

try: epochs = int(sys.argv[1])
except: epochs = 50

   
# Convolutional Neural Network for CIFAR-10 dataset
from keras.datasets import cifar10


def send_results_via_mail(filename):
	recipients = ['carmelrab@gmail.com','amirlivne2@gmail.com']
	emaillist = [elem.strip().split(',') for elem in recipients]
	msg = MIMEMultipart()
	#msg['Subject'] = str(sys.argv[1])
	msg['Subject'] = 'project A test results'
	msg['From'] = 'ProjectA.results@gmail.com'
	msg['Reply-to'] = 'ProjectA.results@gmail.com'
	 
	msg.preamble = 'Multipart massage.\n'
	 
	part = MIMEText("Hi, please find the attached file")
	msg.attach(part)

	#filename = str(sys.argv[2])
	#filename = 'D:\\TECHNION\\projectA\\tasks.txt'
	part = MIMEApplication(open(filename,"rb").read())

	part.add_header('Content-Disposition', 'attachment', filename=filename)
	msg.attach(part)
	 
	server = smtplib.SMTP("smtp.gmail.com:587")
	server.ehlo()
	server.starttls()
	server.login("ProjectA.results@gmail.com", "carmelamir")
	 
	server.sendmail(msg['From'], emaillist , msg.as_string())
    
    
input_shape=(3, 32, 32)
num_classes = 10


# Define the model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# Compile the model
model.compile(optimizer=SGD(lr=0.05, momentum=0.0, decay=0.0, nesterov=False), 
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
   
batch_size = 250
max_batch_num = int(num_train/batch_size)
X = []
Y = []
for i in range(int(num_train/batch_size)):
    X.append(x_train[(i * batch_size):((i+1) * batch_size - 1),:,:,:])
    Y.append(y_train[(i * batch_size):((i+1) * batch_size - 1),:])

print('Done proccesseing!')


test_lossL = []
accuracyL = []

test_loss, accuracy = model.evaluate(x_test,y_test)
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
    test_loss, accuracy = model.evaluate(x_test,y_test)
    test_lossL.append(test_loss)
    accuracyL.append(accuracy)


print('Done training!')
   
with open('.//baseline_results.log', 'wb') as f:
    pickle.dump([test_lossL, accuracyL, timestamp], f)
print('Dumped results')

fig = plt.figure()
plt.subplot(3, 1, 1)
plt.plot(test_lossL)
plt.title('test loss')
plt.ylabel('loss')

plt.subplot(3, 1, 2)
plt.plot(accuracyL)
plt.title('test accuracy')
plt.ylabel('accuracy [%]')
       
plt.subplot(3, 1, 3)
plt.stem(timestamp)
plt.title('time per epoch')
plt.ylabel('time [sec]')
plt.xlabel('epoch')
fig.savefig('.//baseline_results.png')
#plt.show()

send_results_via_mail('baseline_results.png')
send_results_via_mail('baseline_results.log')

print('Sent results to your mail. GOODBYE!!!')
