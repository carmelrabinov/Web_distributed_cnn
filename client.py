#!/usr/bin/env python
import pika
import time
import sys
import json
import numpy as np

# to aviod warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")  

def weightsDiff(W1,W2):
    diff = W1
    i=0
    for l1,l2 in zip(W1,W2):
        diff[i] = l2-l1
        i=i+1
    return diff


###### tmporal - untill implementing batch data connection

from keras.datasets import mnist
import keras
num_classes = 10

# the data, shuffled and split between train and test sets
(x_train, y_train), (_, _) = mnist.load_data()
# input image dimensions
img_rows, img_cols = 28, 28
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_train /= 255
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)

batch_size = 100
global X, Y
X=[]
Y=[]

for i in range(600):
    X.append(x_train[i:(i+100),:,:,:])
    Y.append(y_train[i:(i+100),:])


###### end of temporal part

name = sys.argv[1]

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))

channel = connection.channel()


channel.queue_declare('model_build '+name)
channel.queue_bind(queue='model_build '+name,
                   exchange='pika',
                   routing_key='model_build '+name)

# new_client annoncment
ready_msg = dict(name=name, device = 'pc')
channel.basic_publish(exchange='pika',
                      routing_key='new_client',
                      body=json.dumps(ready_msg))


channel.queue_bind(queue='requests', 
                   exchange='pika', 
                   routing_key='requests')

global model

#def callback(ch, method, properties, body):
#    body = json.loads(body.decode('utf-8'))
#    data = body['data']
#    print("recieved {} from server [{}]".format(data, name))
#
#    ns = {}
#    exec(body['fn'], ns)
#    the_script = ns['the_script']
##    time.sleep(2)
#
#    result = the_script(name, data)
#    print('result = ',result)
#    channel.basic_publish(exchange='pika',
#                          routing_key='results',
#                          body=str(result))
#
#    channel.basic_ack(method.delivery_tag)

# this callback accure every time it train on batch 
def train_batch_callback(ch, method, properties, body):
    body = json.loads(body.decode('utf-8'))
    weights = list(np.asarray(lis,dtype=np.float32) for lis in body['weights'])
    batch_num = body['batch_num']
    print("recieved weights from server")
    
    
    model.set_weights(weights)
    train_loss = model.train_on_batch(X[batch_num], Y[batch_num])
    new_weights = model.get_weights()
    diff_weights = weightsDiff(weights,new_weights)
    
    data = dict(weights = list(np.array(arr).tolist() for arr in diff_weights),train_loss=np.array(train_loss).tolist(), name =name )
    channel.basic_publish(exchange='pika',
                          routing_key='results',
                          body=json.dumps(data))
    print(" [x] Sent diff_weights to server")

    channel.basic_ack(method.delivery_tag)
    channel.basic_publish(exchange='pika',
                      routing_key='ready',
                      body=json.dumps(ready_msg))

# this callback accure when the connection between client and server is establish 
# and responssible for building the nn model
def build_model_callback(ch, method, properties, body):
    body = json.loads(body.decode('utf-8'))
    print("recieved model from server")

    ns = {}
    exec(body['fn'], ns)
    build_model = ns['build_model']
    
    global model
    model = build_model()

    channel.basic_ack(method.delivery_tag)
    channel.basic_publish(exchange='pika',
                      routing_key='ready',
                      body=json.dumps(ready_msg))


channel.basic_consume(build_model_callback,
                      queue='model_build '+name,
                      no_ack=False)


channel.basic_consume(train_batch_callback,
                      queue='requests',
                      no_ack=False)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()

