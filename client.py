#!/usr/bin/env python
import pika
import time
import sys
import json
import numpy as np
from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")  


name = sys.argv[1]

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))

channel = connection.channel()


channel.queue_declare('model_build'+name)
channel.queue_bind(queue='model_build'+name,
                   exchange='pika',
                   routing_key='model_build'+name)
print('model_build'+name)

# new_client annoncment
ready_msg = dict(name=name, device = 'pc')
channel.basic_publish(exchange='pika',
                      routing_key='new_client',
                      body=json.dumps(ready_msg))


channel.basic_publish(exchange='pika',
                      routing_key='ready',
                      body=str(ready_msg))

channel.queue_bind(queue='requests', 
                   exchange='pika', 
                   routing_key='requests')



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


def callback(ch, method, properties, body):
    body = json.loads(body.decode('utf-8'))
    data = list(np.asarray(lis) for lis in body['weights']) 
    print("recieved request from server")

    result = np.sum(np.sum(data))
    print('result = ',result)
    channel.basic_publish(exchange='pika',
                          routing_key='results',
                          body=str(result))
    channel.basic_ack(method.delivery_tag)
    channel.basic_publish(exchange='pika',
                      routing_key='ready',
                      body=str(ready_msg))


def callback2(ch, method, properties, body):
    body = json.loads(body.decode('utf-8'))
    print("recieved model from server")

    ns = {}
    exec(body['fn'], ns)
    build_model = ns['build_model']
#    time.sleep(2)

    model = build_model()
    model.summary()

    channel.basic_ack(method.delivery_tag)


channel.basic_consume(callback2,
                      queue='model_build'+name,
                      no_ack=False)


channel.basic_consume(callback,
                      queue='requests',
                      no_ack=False)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()