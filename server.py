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

# to aviod warnings
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

    
def build_model():
    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
    from keras import backend as K
    
    input_shape = (1, 28, 28)
    num_classes = 10
    
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
    
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

#    weights = model.get_weights()
#    weights_list = list(np.array(arr).tolist() for arr in weights)
#    weights_json=json.dumps(weights_list)
#    data = dict(weights=weights_list)
#    test = json.dumps(data)
#    
#    body = json.loads(test)
#    data = body['weights']
#
#    reconstract = list(np.asarray(lis) for lis in data)
    
    return model



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


###### tmporal - untill implementing batch data connection
from keras.datasets import mnist
import keras
num_classes = 10

# the data, shuffled and split between train and test sets
(_, _), (x_test, y_test) = mnist.load_data()
# input image dimensions
img_rows, img_cols = 28, 28
if K.image_data_format() == 'channels_first':
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_test = x_test.astype('float32')
x_test /= 255
# convert class vectors to binary class matrices
y_test = keras.utils.to_categorical(y_test, num_classes)

###### end of temporal part



def send_model(client_name):
    fn_txt = "".join(inspect.getsourcelines(build_model)[0])
    data = dict(fn=fn_txt)
    channel.basic_publish(exchange='pika',
                          routing_key='model_build '+client_name,
                          body=json.dumps(data))
    print(" [x] Sent nn model to {}".format(client_name))


def send_weights(weights,batch_num):
    data = dict(weights = list(np.array(arr).tolist() for arr in weights),batch_num=batch_num)
    channel.basic_publish(exchange='pika',
                          routing_key='requests',
                          body=json.dumps(data))
    print(" [x] Sent weights calculation request")


def recieved_results(m, body):
    channel.basic_ack(m.delivery_tag)
    print('[x] got results: ', body)


############### main ##################
try:
    mode = sys.argv[1]
except:
    mode = None
        
if mode=='debug':
    model = build_model()
    model.summary()
elif mode=='train':
    model = build_model()
    print('Server setup done')
    max_batch_num = 600
    batch_num = 0
    on_epoch_end = False
    test_loss = [model.test_on_batch(x_test,y_test)]
    timestamp = [time.clock()]
    while True:
        
        # check for new client and respond if exist
        m, _, body = channel.basic_get(queue='new_client', no_ack=True)
        if m:
            data = json.loads(body.decode('utf-8'))
            client_name = data['name']
            send_model(client_name)

        # check reade queue and send current weights and train batch if exist
        m, _, body = channel.basic_get(queue='ready', no_ack=True)
        if m and not on_epoch_end:
            data = json.loads(body.decode('utf-8'))
            print(' [x] recieved ready request from {}'.format(data['name']))
            weights = model.get_weights()
            send_weights(weights,batch_num)
            batch_num = batch_num + 1
            if batch_num == max_batch_num:  # epoch end (note that it doesnt mean that all results came back)
                batch_num=0
                on_epoch_end = True

        # check results queue and update model weights if exist
        m, _, body = channel.basic_get(queue='results', no_ack=False)
        if m:
            data = json.loads(body.decode('utf-8'))
            print(' [x] recieved diff_weights from {} with loss: {}'.format(data['name'],data['train_loss']))
            diff_weights = list(np.asarray(lis, dtype = np.float32) for lis in data['weights'])
            weights = model.get_weights()
            model.set_weights(weightsAdd(weights,diff_weights))
        elif on_epoch_end:  # meaning epoch has ended and got all results back
            test_loss.append(model.test_on_batch(x_test,y_test))
            timestamp.append(time.clock())
            with open('C:\\Users\\carmelr\\projectA\\results.log', 'wb') as f:
                pickle.dump([test_loss, timestamp], f)
            on_epoch_end = False

            

#            recieved_results(m, body)
    #    send(randint(1, 100))
#        time.sleep(5)


