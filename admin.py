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


global model
x_test = None
y_test = None


## this callback accure every time it train on batch 
#def train_batch_callback(ch, method, properties, body):
#    if setup:
#        # extracting weights from json format
#        body = json.loads(body.decode('utf-8'))
#        weights = list(np.asarray(lis,dtype=np.float32) for lis in body['weights'])
#        batch_num = body['batch_num']
##        print(" [x] recieved weights from server")
#    
#        # updating model weights
#        global model
#        model.set_weights(weights)
#        
#        # training and calculating weights diff
#        global X, Y
#        train_loss = model.train_on_batch(X[batch_num], Y[batch_num])
#        new_weights = model.get_weights()
#        diff_weights = weightsDiff(weights,new_weights)
#        
#        # sending weights diff to server
#        data = dict(weights = list(np.array(arr).tolist() for arr in diff_weights),train_loss=np.array(train_loss).tolist(), name =name )
#        channel.basic_publish(exchange='pika',
#                              routing_key='results',
#                              body=json.dumps(data))
#        print(' [x] Sent batch {} diff_weights to server'.format(batch_num))
#        channel.basic_ack(method.delivery_tag)
#        
#        # sending ready msg to server
#        channel.basic_publish(exchange='pika',
#                          routing_key='ready',
#                          body=json.dumps(ready_msg))

# this callback accure every time it train on batch 
def test_callback(ch, method, properties, body):
    if setup:
        # extracting weights from json format
        body = json.loads(body.decode('utf-8'))
        weights = list(np.asarray(lis,dtype=np.float32) for lis in body['weights'])
        batch_num = body['batch_num']
#        print(" [x] recieved weights from server")
    
        # updating model weights
        global model
        model.set_weights(weights)
        
        # testing current net weights
        global x_test, y_test
        test_loss, accuracy = model.evaluate(x_test, y_test)
        print(' [x] finished batch {} with test loss {}, accuracy {}'.format(batch_num, test_loss, accuracy))
#        new_weights = model.get_weights()
#        diff_weights = weightsDiff(weights,new_weights)
#        
#        # sending weights diff to server
#        data = dict(weights = list(np.array(arr).tolist() for arr in diff_weights),train_loss=np.array(train_loss).tolist(), name =name )
#        channel.basic_publish(exchange='pika',
#                              routing_key='results',
#                              body=json.dumps(data))
#        print(' [x] Sent batch {} diff_weights to server'.format(batch_num))
        channel.basic_ack(method.delivery_tag)
        
#        # sending ready msg to server
#        channel.basic_publish(exchange='pika',
#                          routing_key='ready',
#                          body=json.dumps(ready_msg))


# this callback accure when the connection between client and server is establish 
# and responssible for building the nn model
def build_model_callback(ch, method, properties, body):

    body = json.loads(body.decode('utf-8'))
    print(" [x] recieved model from server")

    ns = {}
    exec(body['fn'], ns)
    build_model = ns['build_model']
    dataset = body['dataset']
    
    global model, x_test, y_test
    (model, x_test, y_test) = build_model(dataset = dataset, mode = 'server')

    channel.basic_ack(method.delivery_tag)
#    channel.basic_publish(exchange='pika',
#                      routing_key='ready',
#                      body=json.dumps(ready_msg))
    global setup
    setup = True

############# main #################

   
try:
    host = sys.argv[1]
except:
    host = 'localhost'
 
name = 'admin'
setup = False

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=str(host)))

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


channel.queue_bind(queue='admin', 
                   exchange='pika', 
                   routing_key='admin')


channel.basic_consume(build_model_callback,
                      queue='model_build '+name,
                      no_ack=False)

channel.basic_consume(test_callback,
                      queue='admin',
                      no_ack=False)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()


