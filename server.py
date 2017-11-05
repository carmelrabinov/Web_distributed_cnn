#!/usr/bin/env python
import pika
import sys
from random import randint
import numpy as np
import json
import inspect
import time
try: import cPickle as pickle
except: import pickle

# to aviod TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")  

def weightsAdd(W,W_diff):
    add = W
    i=0
    for l1,l2 in zip(W,W_diff):
        add[i] = l1+l2
        i=i+1
    return add

def build_model(dataset, mode):
    
    import keras
    from keras.models import Sequential
    from keras.layers import Activation, Flatten, Dense, Dropout, Conv2D, MaxPooling2D
    from keras import backend as K

    # Convolutional Neural Network for MNIST dataset
    if dataset == 'mnist':
        from keras.datasets import mnist

        input_shape = (1, 28, 28)
        num_classes = 10
        
        # Define the model
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))
        
        # Compile the model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adadelta',
                      metrics=['accuracy'])
    
   
    # Convolutional Neural Network for CIFAR-10 dataset
    elif dataset == 'cifar10':
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
        
    # load data
    if mode == 'client' and dataset == 'cifar10':
        # the data, shuffled and split between train and test sets
        (x, y), (_, _) = cifar10.load_data()
    elif mode == 'server' and dataset == 'cifar10':
        (_, _), (x, y) = cifar10.load_data()
    elif mode == 'client' and dataset == 'mnist':
        # the data, shuffled and split between train and test sets
        (x, y), (_, _) = mnist.load_data()
    elif mode == 'server' and dataset == 'mnist':
        (_, _), (x, y) = mnist.load_data()

    # preproccessing data
    if dataset == 'cifar10':
        num_train, img_channels, img_rows, img_cols =  x.shape
    if dataset == 'mnist':
        num_train, img_rows, img_cols =  x.shape
        img_channels = 1
    
    x = x.reshape(num_train, img_channels, img_rows, img_cols)
    input_shape = (img_channels, img_rows, img_cols)
    x = x.astype('float32')/255
    # convert class vectors to binary class matrices
    y = keras.utils.to_categorical(y, num_classes)
       
    if mode == 'client':
        batch_size = 100
        X = []
        Y = []
        for i in range(int(num_train/batch_size)):
            X.append(x[(i * batch_size):((i+1) * batch_size - 1),:,:,:])
            Y.append(y[(i * batch_size):((i+1) * batch_size - 1),:])
        return (model, X, Y)

    elif mode == 'server':
        global x_test, y_test
        y_test = y
        x_test = x

        return model


def send_model(client_name, dataset):
    fn_txt = "".join(inspect.getsourcelines(build_model)[0])
    data = dict(fn=fn_txt, dataset = dataset)
    channel.basic_publish(exchange='pika',
                          routing_key='model_build '+client_name,
                          body=json.dumps(data))
    print(" [x] Sent cnn model to {}".format(client_name))


def send_weights(weights,batch_num):
    data = dict(weights = list(np.array(arr).tolist() for arr in weights),batch_num=batch_num)
    channel.basic_publish(exchange='pika',
                          routing_key='requests',
                          body=json.dumps(data))
#    print(" [x] Sent weights calculation request")


def recieved_results(m, body):
    channel.basic_ack(m.delivery_tag)
    print('[x] got results: ', body)


############### main ##################


# setting up the connection
print('Server is setting up...')
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
channel = connection.channel()


channel.exchange_declare(exchange='pika',
                         exchange_type='direct',
                         durable=False,
                         auto_delete=True)

channel.queue_delete(queue='results')
channel.queue_delete(queue='requests')
channel.queue_delete(queue='ready')
channel.queue_delete(queue='new_client')


# this queu holds the results (diff weights) sent from the clients to the server
channel.queue_declare('results')
channel.queue_bind(queue='results',
                   exchange='pika',
                   routing_key='results')

# this queu holds the server requests (a batch and current weights) sent from the server to the clients
channel.queue_declare('requests')
channel.queue_bind(queue='requests',
                   exchange='pika',
                   routing_key='requests')


# this queu holds the ready statments sent by clients to the server who are ready to train
channel.queue_declare('ready')
channel.queue_bind(queue='ready',
                   exchange='pika',
                   routing_key='ready')

# this queu holds the ready statments sent by clients to the server who are ready to train
channel.queue_declare('new_client')
channel.queue_bind(queue='new_client',
                   exchange='pika',
                   routing_key='new_client')


try:
    mode = sys.argv[1]
except:
    mode = None

try:
    dataset = sys.argv[2]
except:
    dataset = 'mnist'


        
if mode=='debug':
    model = build_model(dataset, mode = 'server')
    model.summary()
elif mode=='train':
    model = build_model(dataset, mode = 'server')
    print('Server setup done')
    max_batch_num = 500 # sould be 600 for mnist
    batch_num = 0
    batch_count = 0
    epoch = 0
    test_lossL = []
    accuracyL = []
    test_loss, accuracy = model.test_on_batch(x_test,y_test)
    test_lossL.append(test_loss)
    accuracyL.append(accuracy)
    start_time = time.time()
    timestamp = [0]
    print(' [x] initial training with test loss: {}, accuracy: {}'.format(test_loss, accuracy))
    while True:
        
        # check for new client and respond if exist
        m, _, body = channel.basic_get(queue='new_client', no_ack=True)
        if m:
            data = json.loads(body.decode('utf-8'))
            client_name = data['name']
            send_model(client_name, dataset)

        # check reade queue and send current weights and train batch if exist
        m, _, body = channel.basic_get(queue='ready', no_ack=True)
        if m:
            data = json.loads(body.decode('utf-8'))
#            print(' [x] recieved ready request from {}'.format(data['name']))
            weights = model.get_weights()
            send_weights(weights,batch_num)
            batch_num += 1
            if batch_num == max_batch_num:  # epoch end (note that it doesnt mean that all results came back)
                batch_num=0

        # check results queue and update model weights if exist
        m, _, body = channel.basic_get(queue='results', no_ack=False)
        if m:
            batch_count += 1
            data = json.loads(body.decode('utf-8'))
            print(' [x] recieved batch {} diff_weights from {} with loss: {}'.format(batch_count, data['name'],data['train_loss']))
            diff_weights = list(np.asarray(lis, dtype = np.float32) for lis in data['weights'])
            weights = model.get_weights()
            model.set_weights(weightsAdd(weights,diff_weights))
            channel.basic_ack(m.delivery_tag)
            if batch_count % 50 == 0:
                test_loss, accuracy = model.test_on_batch(x_test,y_test)
                test_lossL.append(test_loss)
                accuracyL.append(accuracy)
                timestamp.append(time.time() - start_time)
                print(' [x] batch {} with test loss: {}, accuracy: {}, time: {}'.format(batch_count, test_loss, accuracy, timestamp[-1]))
                with open('C:\\Users\\carmelr\\projectA\\results.log', 'wb') as f:
                    pickle.dump([test_lossL, accuracyL, timestamp], f)
            if batch_count == max_batch_num:  # meaning epoch has ended and got all results back
                batch_count = 0
                epoch += 1
                print(' [x] finished epoch {}'.format(epoch))


            

#            recieved_results(m, body)
    #    send(randint(1, 100))
#        time.sleep(5)


