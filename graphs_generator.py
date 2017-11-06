#!/usr/bin/env python
import pika
import time
import sys
import json
import numpy as np
import csv
try: import cPickle as pickle
except: import pickle

# to aviod warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")  



# initial global variables:
global model
X = []
Y = []
test_batches_num = 0
start_time = time.time()
setup = False
Loss=[]
Acc=[]
Time=[]

# this callback accure when the connection between client and server is establish 
# and responssible for building the nn model
def build_model_callback(ch, method, properties, body):
    print('initialising model')
   
    
    body = json.loads(body.decode('utf-8'))
    ns = {}
    exec(body['fn'], ns)
    build_model = ns['build_model']
    dataset = body['dataset']
    
    global model, X, Y, test_batches_num
    (model, X, Y, test_batches_num) = build_model(dataset = dataset, mode = 'test')
    print ('test batches num is: ',test_batches_num)
    channel.basic_ack(method.delivery_tag)
    
    global setup
    setup = True    
    print('logger setup done')

# =============================================================================
# def build_model_callback2(ch, method, properties, body):
#     print('debug')
# =============================================================================
    
def test_batch_callback(ch, method, properties, body):
    if setup:
        print('got weights from server')

        # extracting weights from json format
        body = json.loads(body.decode('utf-8'))
        weights = list(np.asarray(lis,dtype=np.float32) for lis in body['weights'])
#        print(" [x] recieved weights from server")
    
#        channel.basic_ack(method.delivery_tag)

    
        # updating model weights
        global model
        model.set_weights(weights)
        
        #testing over all batches
        tot_loss = 0
        tot_accuracy = 0
        print('start training')
#        for i in range (test_batches_num):
        for i in range (test_batches_num):
            loss, accuracy = model.test_on_batch(X[i],Y[i])
            tot_loss += loss
            tot_accuracy += accuracy
        print('finished training')

        #calculating avarage values and print to log 
        global Loss, Acc, Time
        Loss.append(tot_loss / test_batches_num)
        Acc.append(tot_accuracy / test_batches_num)
        Time.append(body['time'])
        
        with open('C:\\Users\\amirli\\Desktop\\amir\\results.log', 'wb') as f:
            pickle.dump([Loss, Acc, Time], f)
        print('dumped data to pickle')
        




############# main #################

setup = False

try:
    host = sys.argv[1]
except:
    host = 'localhost'
    
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host=str(host)))

channel = connection.channel()


channel.queue_bind(queue='model_build logger', 
                   exchange='pika', 
                   routing_key='model_build logger')


channel.queue_bind(queue='logger', 
                   exchange='pika', 
                   routing_key='logger')



channel.basic_consume(build_model_callback,
                      queue='model_build logger',
                      no_ack=False)


channel.basic_consume(test_batch_callback,
                      queue='logger',
                      no_ack=True)


        
        
print(' [~] Welcome to "Logger" - your statistics and graphs generator!')
channel.start_consuming()

