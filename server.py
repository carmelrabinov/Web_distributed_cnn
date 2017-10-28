#!/usr/bin/env python
import pika
import sys
from random import randint
import numpy as np
import json
import inspect
import time
from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")  



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


# the method to send

def the_script(name, n):
    res = n * 2
#    return "{0} * 2 = {1} [{2}]".format(n, res, name)
    return res


def send(num):
    fn_txt = "".join(inspect.getsourcelines(the_script)[0])
    data = dict(fn=fn_txt, data=num)

    channel.basic_publish(exchange='pika',
                          routing_key='requests',
                          body=json.dumps(data))
    print(" [x] Sent '%s'" % json.dumps(data['data']))


def send_model(client_name):
    fn_txt = "".join(inspect.getsourcelines(build_model)[0])
    data = dict(fn=fn_txt)
    channel.basic_publish(exchange='pika',
                          routing_key='model_build'+client_name,
                          body=json.dumps(data))
    print(" [x] Sent model to {}".format(client_name))


def send_weights(weights):
    data = dict(weights = list(np.array(arr).tolist() for arr in weights))
    channel.basic_publish(exchange='pika',
                          routing_key='requests',
                          body=json.dumps(data))
    print(" [x] Sent weights calculation request")


def recieved(m, body):
    channel.basic_ack(m.delivery_tag)

def recieved_results(m, body):
    channel.basic_ack(m.delivery_tag)
    print('got results: ', body)


try:
    mode = sys.argv[1]
except:
    mode = None
        
if mode=='debug':
    model = build_model()
    model.summary()
elif mode=='train':
    model = build_model()
    print('Setup done')
    while True:
        m, _, body = channel.basic_get(queue='new_client', no_ack=True)
        if m:
            data = json.loads(body.decode('utf-8'))
            client_name = data['name']
            send_model(client_name)

#        m, _, body = channel.basic_get(queue='ready', no_ack=True)
#        if m:
#            print (body)
#            weights = model.get_weights()
#            send_weights(weights)
#        m, _, body = channel.basic_get(queue='results', no_ack=False)
#        if m:
#            recieved_results(m, body)
#    #    send(randint(1, 100))
        time.sleep(5)


