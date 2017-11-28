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

from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from smtplib import SMTP
import smtplib

import matplotlib.pyplot as plt

# to aviod TF warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras import backend as K
if K.backend()=='tensorflow':
    K.set_image_dim_ordering("th")  

def send_results_via_mail(filename):
	recipients = ['carmelrab@gmail.com','amirlivne2@gmail.com']
	emaillist = [elem.strip().split(',') for elem in recipients]
	msg = MIMEMultipart()
	#msg['Subject'] = str(sys.argv[1])
	msg['Subject'] = 'project A test results'
	msg['From'] = 'ProjectA.results@gmail.com'
	msg['Reply-to'] = 'ProjectA.results@gmail.com'
	 
	msg.preamble = 'Multipart massage.\n'
	 
	part = MIMEText("Hi, please find the attached file")
	msg.attach(part)

	#filename = str(sys.argv[2])
	#filename = 'D:\\TECHNION\\projectA\\tasks.txt'
	part = MIMEApplication(open(filename,"rb").read())

	part.add_header('Content-Disposition', 'attachment', filename=filename)
	msg.attach(part)
	 
	server = smtplib.SMTP("smtp.gmail.com:587")
	server.ehlo()
	server.starttls()
	server.login("ProjectA.results@gmail.com", "carmelamir")
	 
	server.sendmail(msg['From'], emaillist , msg.as_string())

def results_calculations(model,weightsL,timestamp,filename):
    lossL = []
    accuracyL = []
    for weights in weightsL:
        model.set_weights(weights)
        loss, accuracy = model.evaluate(x_test, y_test)
        lossL.append(loss)
        accuracyL.append(accuracy)
        print('loss: {}, accuracy: {}'.format(loss, accuracy))
    
    with open(filename+'.log', 'wb') as f2:
        pickle.dump([lossL, accuracyL, timestamp], f2)
    
    send_results_via_mail(filename+'.log')
    
    timestamp[1:] = np.asarray(timestamp[1:]) - np.asarray(timestamp[0:-1])

    fig = plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(lossL)
    plt.title('test loss')
    plt.ylabel('loss')
    
    plt.subplot(3, 1, 2)
    plt.plot(accuracyL)
    plt.title('test accuracy')
    plt.ylabel('accuracy [%]')
	   
    plt.subplot(3, 1, 3)
    plt.stem(timestamp)
    plt.title('time per epoch')
    plt.ylabel('time [sec]')
    plt.xlabel('epoch')
    fig.savefig(filename+'.png')
    
    send_results_via_mail(filename+'.png')
 #   plt.show()

def build_model(dataset, mode):
    
    import keras
    from keras.models import Sequential
    from keras.layers import Activation, Flatten, Dense, Dropout, Conv2D, MaxPooling2D
    from keras.optimizers import SGD

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
        model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        
        # Compile the model
        model.compile(optimizer=SGD(lr=0.05, momentum=0.0, decay=0.0, nesterov=False), 
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
        batch_size = 250
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
#    if logPrint: print(" [x] Sent cnn model to {}".format(client_name))
    print(" [x] Sent cnn model to {}".format(client_name))


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

def got_results(m, body):
    
    global timeL, weightsL, curr_weights, batch_count, epoch
 
    batch_count += 1
    recieved_weights(body)            
    channel.basic_ack(m.delivery_tag)
            
    # run on test mode: calculate weights every test batched instead of every epoch
    if test and batch_count % test == 0:
        if noAdmin:
            timeL.append(time.time() - start_time)
            weightsL.append(copy.deepcopy(curr_weights))
            with open('./test_results/'+fn+'.pkl', 'wb') as f:
                pickle.dump([weightsL, timeL], f)
#                    results_calculations(model,weightsL,timeL,'.//test_results//'+fn)
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
            with open('./test_results/'+fn+'.pkl', 'wb') as f:
                pickle.dump([weightsL, timeL], f)
        else:
            send_test_weights(curr_weights, batch_count, time.time() - start_time)


############### main ##################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-dataset', type=str, default='cifar10')
    parser.add_argument('-logPrint', action='store_true')
    parser.add_argument('-noAdmin', action='store_true')
    parser.add_argument('-fn', type=str, default='weights')
    parser.add_argument('-test', type=int, default=0)
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-shuffle', action='store_true')
    parser.add_argument('-host', type=str, default='132.68.60.181')

    parser.parse_args(namespace=sys.modules['__main__'])
    

    # setting up the connection
    print('Server is setting up...')
    credentials = pika.PlainCredentials('admin', 'admin')
    parameters = pika.ConnectionParameters(host = host,
				port = 5672,
    				virtual_host = '/',
				credentials = credentials)    
    connection = pika.BlockingConnection(parameters)
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
        weightsL.append(copy.deepcopy(curr_weights))
    print('Server setup done\n')
    print('=== model stats ===')
    print('params: {}'.format(model.count_params())) 
    print('dataset: {}'.format(dataset))
    
    global max_batch_num,batch_num, batch_count, epoch, start_time
    
    max_batch_num = 200 # sould be 600 for mnist
    batch_num = 0
    batch_count = 0
    epoch = 0
    
    start_time = time.time()
  
    s = np.arange(max_batch_num)
            
    while True:      
        
        # check if train is ended
        if epoch == epochs:
            results_calculations(model,weightsL,timeL,'.//test_results//'+fn)
            exit()        
        
        # check results queue and update curr weights if exist
        m3, _, body3 = channel.basic_get(queue='results', no_ack=False)
        while m3:
            got_results(m3, body3)
            m3, _, body3 = channel.basic_get(queue='results', no_ack=False)
     
        # check for new client and respond if exist
        m1, _, body1 = channel.basic_get(queue='new_client', no_ack=True)
        if m1:
            data = json.loads(body1.decode('utf-8'))
            client_name = data['name']
            send_model(client_name, dataset)
    
        # check ready queue and send current weights and train batch if exist
        m2, _, body2 = channel.basic_get(queue='ready', no_ack=True)
        if m2:
            m3, _, body3 = channel.basic_get(queue='results', no_ack=False)
            if m3:
                got_results(m3, body3)

#            data = json.loads(body2.decode('utf-8'))
#            print(' [x] recieved ready request from {}'.format(data['name']))
            batch_num += 1
            if batch_num == max_batch_num:  # epoch end (note that it doesnt mean that all results came back)
                batch_num=0
                if shuffle:
                    np.random.shuffle(s)  # shuffle the order of the batches in each epoch
            send_weights(curr_weights,int(s[batch_num]))
#            m2, _, body2 = channel.basic_get(queue='ready', no_ack=True)
  
