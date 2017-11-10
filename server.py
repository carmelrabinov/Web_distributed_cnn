#!/usr/bin/env python
import pika
import sys
import numpy as np
import json
import inspect
import time
import argparse
try: import cPickle as pickle
except: import pickle
import copy


# to aviod TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")  

#def weightsAdd(W,W_diff):
#    i=0
#    for l1,l2 in zip(W,W_diff):
#        W[i] = l1+l2
#        i=i+1

def build_model(dataset, mode):
    
    import keras
    from keras.models import Sequential
    from keras.layers import Activation, Flatten, Dense, Dropout, Conv2D, MaxPooling2D

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
        model.compile(optimizer='SGD', 
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

        return (model, x_test, y_test)


def send_model(client_name, dataset):
    fn_txt = "".join(inspect.getsourcelines(build_model)[0])
    data = dict(fn=fn_txt, dataset = dataset)
    channel.basic_publish(exchange='pika',
                          routing_key='model_build '+client_name,
                          body=json.dumps(data))
    if logPrint: print(" [x] Sent cnn model to {}".format(client_name))


def send_weights(weights,batch_num):
    data = dict(weights = list(np.array(arr).tolist() for arr in weights),batch_num=batch_num)
    channel.basic_publish(exchange='pika',
                          routing_key='requests',
                          body=json.dumps(data))
    if logPrint: print(" [x] Sent weights calculation request")

def recieved_weights(body):
        data = json.loads(body.decode('utf-8'))
        if logPrint: print(' [x] recieved batch {} diff_weights from {} with loss: {}'.format(batch_count, data['name'],data['train_loss']))
        diff_weights = list(np.asarray(lis, dtype = np.float32) for lis in data['weights'])
        
        global curr_weights
        for i in range(len(curr_weights)):
            curr_weights[i] += diff_weights[i]

#        weightsAdd(curr_weights,diff_weights)


def send_test_weights(weights, batch_num, time):
    data = dict(weights = list(np.array(arr).tolist() for arr in weights),batch_num = batch_num, time = time)
    channel.basic_publish(exchange='pika',
                          routing_key='admin',
                          body=json.dumps(data))
#    print(" [x] Sent weights calculation request")



############### main ##################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dataset', type=str, default='cifar10')
    parser.add_argument('-logPrint', action='store_true')
    parser.add_argument('-noAdmin', action='store_true')
    parser.add_argument('-fn', type=str, default='weights')
    parser.add_argument('-test', type=int, default=0)
    parser.parse_args(namespace=sys.modules['__main__'])
    


    # setting up the connection
    print('Server is setting up...')
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    
    
    channel.exchange_declare(exchange='pika',
                             exchange_type='direct',
                             durable=False,
                             auto_delete=True)
    
    channel.queue_delete(queue='requests')
    channel.queue_delete(queue='ready')
    channel.queue_delete(queue='new_client')
    channel.queue_delete(queue='admin')
    channel.queue_delete(queue='results')
    channel.queue_delete(queue='model_build admin')
    for i in range(10):
        channel.queue_delete(queue='model_build '+str(i))
    
    
    
    
    # this queu holds the server requests (a batch and current weights) sent from the server to the clients
    channel.queue_declare('requests')
    channel.queue_bind(queue='requests',
                       exchange='pika',
                       routing_key='requests')
    
    # this queu holds the results (diff weights) sent from the clients to the server
    channel.queue_declare('results')
    channel.queue_bind(queue='results',
                       exchange='pika',
                       routing_key='results')

    
    if not noAdmin:
        # this queu holds the server test requests (current weights) sent from the server to the admin for test
        channel.queue_declare('admin')
        channel.queue_bind(queue='admin',
                           exchange='pika',
                           routing_key='admin')
    
    
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
    
    
          
    (model, _, _) = build_model(dataset, mode = 'server')
    global curr_weights
    curr_weights = model.get_weights()
    if noAdmin:
        weightsL = []
        timeL = [0]
        weightsL.append(curr_weights)
    print('Server setup done\n')
    print('=== model stats ===')
    print('params: {}'.format(model.count_params())) 
    print('dataset: {}'.format(dataset))

    max_batch_num = 500 # sould be 600 for mnist
    batch_num = 0
    batch_count = 0
    epoch = 0
    
    start_time = time.time()
    
    while True:      
        # check for new client and respond if exist
        m, _, body = channel.basic_get(queue='new_client', no_ack=True)
        if m:
            data = json.loads(body.decode('utf-8'))
            client_name = data['name']
            send_model(client_name, dataset)
    
        # check ready queue and send current weights and train batch if exist
        m, _, body = channel.basic_get(queue='ready', no_ack=True)
        if m:
#            data = json.loads(body.decode('utf-8'))
#            print(' [x] recieved ready request from {}'.format(data['name']))
            batch_num += 1
            if batch_num == max_batch_num:  # epoch end (note that it doesnt mean that all results came back)
                batch_num=0           
            send_weights(curr_weights,batch_num)
#            m, _, body = channel.basic_get(queue='ready', no_ack=True)
  
        # check results queue and update curr weights if exist
        m, _, body = channel.basic_get(queue='results', no_ack=False)
        if m:
            batch_count += 1
            recieved_weights(body)            
            channel.basic_ack(m.delivery_tag)
#            m, _, body = channel.basic_get(queue='results', no_ack=False)
            
            # run on test mode: calculate weights every test batched instead of every epoch
            if test and batch_count % test == 0:
                if noAdmin:
                    timeL.append(time.time() - start_time)
                    weightsL.append(copy.deepcopy(curr_weights))
                    with open('C:\\Users\\carmelr\\projectA\\test_results\\'+fn+'.pkl', 'wb') as f:
                        pickle.dump([weightsL, timeL], f)
                else:
                    send_test_weights(curr_weights, batch_count, time.time() - start_time)
            
            # meaning epoch has ended and got all results back
            elif batch_count == max_batch_num:  
                batch_count = 0
                epoch += 1
                print(' [x] finished epoch {}'.format(epoch))
                if noAdmin:
                    timeL.append(time.time() - start_time)
                    weightsL.append(copy.deepcopy(curr_weights))
                    with open('C:\\Users\\carmelr\\projectA\\test_results\\'+fn+'.pkl', 'wb') as f:
                        pickle.dump([weightsL, timeL], f)
                else:
                    send_test_weights(curr_weights, batch_count, time.time() - start_time)

