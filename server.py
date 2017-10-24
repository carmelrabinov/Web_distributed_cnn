#!/usr/bin/env python
import pika
from random import randint
import json
import inspect
import time

# setting up the connection
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.exchange_declare(exchange='pika',
                         exchange_type='direct',
                         durable=False,
                         auto_delete=True)

channel.queue_declare('results')
channel.queue_bind(queue='results',
                   exchange='pika',
                   routing_key='results')

# the method to send


def the_script(name, n):
    res = n * 2
    return "{0} * 2 = {1} [{2}]".format(n, res, name)


def send(num):
    fn_txt = "".join(inspect.getsourcelines(the_script)[0])
    data = dict(fn=fn_txt, data=num)

    channel.basic_publish(exchange='pika',
                          routing_key='request',
                          body=json.dumps(data))
    print(" [x] Sent '%s'" % json.dumps(data['data']))


def recieved(m, body):
    print(body)
    channel.basic_ack(m.delivery_tag)


while True:
    m, _, body = channel.basic_get(queue='results', no_ack=True)
    if m:
        recieved(m, body)
    send(randint(1, 100))
    time.sleep(1)
