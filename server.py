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

##DEFINES:
SEND_TO_LOGGER = -1

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
    elif (mode == 'server' or mode == 'test') and dataset == 'cifar10':
        (_, _), (x, y) = cifar10.load_data()
    elif mode == 'client' and dataset == 'mnist':
        # the data, shuffled and split between train and test sets
        (x, y), (_, _) = mnist.load_data()
    elif (mode == 'server' or mode == 'test') and dataset == 'mnist':
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
    
    elif mode == 'test':
        print('num of images to test is: ',num_train) #Debug
        #Debug - perform test only on a specific batch
        test_batch_size = 1000
        X = []
        Y = []
        for i in range(int(num_train/test_batch_size)):
            X.append(x[(i * test_batch_size):((i+1) * test_batch_size - 1),:,:,:]) 
            Y.append(y[(i * test_batch_size):((i+1) * test_batch_size - 1),:])
        return (model,X,Y,int(num_train/test_batch_size))


def send_model(client_name, dataset):
    fn_txt = "".join(inspect.getsourcelines(build_model)[0])
    data = dict(fn=fn_txt, dataset = dataset)
    channel.basic_publish(exchange='pika',
                          routing_key='model_build '+client_name,
                          body=json.dumps(data))
    print(" [x] Sent cnn model to {}".format(client_name))

# =============================================================================
# def send_model2(dataset):
#     print('sent 2')
#     fn_txt = "".join(inspect.getsourcelines(build_model)[0])
#     data = dict(fn=fn_txt, dataset = dataset)
#     channel.basic_publish(exchange='pika',
#                           routing_key='debug',
#                           body=json.dumps(data))
# =============================================================================

def send_weights(weights,batch_num,rounting_key):
    data = dict(weights = list(np.array(arr).tolist() for arr in weights),batch_num=batch_num,time=time.time()-start_time)
    channel.basic_publish(exchange='pika',
                          routing_key=rounting_key,
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
channel.queue_delete(queue='model_build logger')
channel.queue_delete(queue='logger')
#channel.queue_delete(queue='debug')


# this queu holds the results (diff weights) sent from the clients to the server
channel.queue_declare('results')
channel.queue_bind(queue='results',
                   exchange='pika',
                   routing_key='results')

# this queu holds the server requests (a batch and current weights) sent from the server to the clients
channel.queue_declare('requests')



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

# this queu is for initilaising the logger
channel.queue_declare('model_build logger')
channel.queue_bind(queue='model_build logger',
                   exchange='pika',
                   routing_key='model_build logger')

# this queue is for updating the logger with the latests nn
channel.queue_declare('logger')
channel.queue_bind(queue='logger',
                   exchange='pika',
                   routing_key='logger')

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
    #initialising the logger:
    send_model('logger',dataset)
    start_time = time.time()
    send_weights(model.get_weights(),0,'logger')

#    send_model2(dataset)

    max_batch_num = 500 # sould be 600 for mnist
    batch_num = 0
    batch_count = 0
    epoch = 0

    print('dataset: {}'.format(dataset))
#    print(' [x] initial training with test loss: {}, accuracy: {}'.format(test_loss, accuracy))
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
            send_weights(weights,batch_num,'requests')
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
            new_weights = weightsAdd(weights,diff_weights)
            model.set_weights(new_weights)
            channel.basic_ack(m.delivery_tag)
#debug            if batch_count % 50 == 0:
            if  batch_count % 5 == 0:
                send_weights(new_weights,0,'logger')
                print('activated logger')
# =============================================================================
#                 with open('C:\\Users\\carmelr\\projectA\\results.log', 'wb') as f:
#                     pickle.dump([test_lossL, accuracyL, timestamp], f)
# =============================================================================
            if batch_count == max_batch_num:  # meaning epoch has ended and got all results back
                batch_count = 0
                epoch += 1
                print(' [x] finished epoch {}'.format(epoch))


            

#            recieved_results(m, body)
    #    send(randint(1, 100))
#        time.sleep(5)


