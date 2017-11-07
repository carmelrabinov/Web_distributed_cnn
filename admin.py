#!/usr/bin/env python
import pika
import sys
import json
import numpy as np
try: import cPickle as pickle
except: import pickle

# to aviod warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")  

# global parameters
global model
x_test = None
y_test = None

test_lossL = []
accuracyL = []
timestamp = [0]


# this callback acure every time a test request on all the test data sends by the server to the admin 
def test_callback(ch, method, properties, body):
    if setup:
        # extracting weights from json format
        body = json.loads(body.decode('utf-8'))
        weights = list(np.asarray(lis,dtype=np.float32) for lis in body['weights'])
        batch_num = body['batch_num']
        time = body['time']

        channel.basic_ack(method.delivery_tag)

#        print(" [x] recieved weights from server")
    
        # updating model weights
        global model
        model.set_weights(weights)
        
        # testing current net weights
        global x_test, y_test
        print(' [x] start testing after batch {} backpropagation'.format(batch_num))
        test_loss, accuracy = model.evaluate(x_test, y_test)
        print(' [x] finished batch {} with test loss {}, accuracy {}'.format(batch_num, test_loss, accuracy))
        
        # save and dump test results
        test_lossL.append(test_loss)
        accuracyL.append(accuracy)
        timestamp.append(time)
        with open('C:\\Users\\carmelr\\projectA\\test_results\\'+resultsFn+'.log', 'wb') as f:
            pickle.dump([test_lossL, accuracyL, timestamp], f)

        
# this callback accure when the connection between client and server is establish 
# and responssible for building the nn model
def build_model_callback(ch, method, properties, body):

    # extracting build_model function from json format
    body = json.loads(body.decode('utf-8'))
    ns = {}
    exec(body['fn'], ns)
    build_model = ns['build_model']
    dataset = body['dataset']
    print(" [x] recieved model from server")

    channel.basic_ack(method.delivery_tag)
   
    # build nn model
    global model, x_test, y_test
    (model, x_test, y_test) = build_model(dataset = dataset, mode = 'server')

    # compute initial test loss and accuracy
    test_loss, accuracy = model.test_on_batch(x_test,y_test)
    test_lossL.append(test_loss)
    accuracyL.append(accuracy)
    print(' [x] initial training with test loss: {}, accuracy: {}'.format(test_loss, accuracy))

    # enable test_callback
    global setup
    setup = True

############# main #################

   
#try:
#    host = sys.argv[1]
#except:
#    host = 'localhost'

try:
    resultsFn = sys.argv[1]
except:
    resultsFn = 'results'

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


