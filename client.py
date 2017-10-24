#!/usr/bin/env python
import pika
import time
import sys
import json

name = 'hello ' + sys.argv[1]

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))

channel = connection.channel()

channel.queue_declare(queue=name)
channel.queue_bind(queue=name, exchange='pika', routing_key='request')


def callback(ch, method, properties, body):
    body = json.loads(body.decode('utf-8'))
    data = body['data']
    print("recieved {} from server [{}]".format(data, name))

    ns = {}
    exec(body['fn'], ns)
    the_script = ns['the_script']
    time.sleep(2)

    result = the_script(name, data)
    channel.basic_publish(exchange='pika',
                          routing_key='results',
                          body=str(result))

    channel.basic_ack(method.delivery_tag)


channel.basic_consume(callback,
                      queue=name,
                      no_ack=False)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()