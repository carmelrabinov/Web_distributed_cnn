# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 20:35:03 2017

@author: carmelr
"""
import sys
import numpy as np
try: import cPickle as pickle
except: import pickle
import matplotlib.pyplot as plt
import argparse
try: import cPickle as pickle
except: import pickle


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
    
    
def build_model():
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    
    from keras import backend as K
    if K.backend()=='tensorflow':
        K.set_image_dim_ordering("th")  
    
    import keras
    from keras.models import Sequential
    from keras.layers import Activation, Flatten, Dense, Dropout, Conv2D, MaxPooling2D
    from keras.optimizers import SGD
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
    
     
    # load and preproccessing data
    (_, _), (x, y) = cifar10.load_data()
    num_train, img_channels, img_rows, img_cols =  x.shape   
    x = x.reshape(num_train, img_channels, img_rows, img_cols)
    input_shape = (img_channels, img_rows, img_cols)
    x = x.astype('float32')/255
    # convert class vectors to binary class matrices
    y = keras.utils.to_categorical(y, num_classes)
    print("built model")   
    return (model, x, y)


#fn = 'weights.pkl'
#noAdmin = True
#baseline = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('fn', help='file name')
    parser.add_argument('-noAdmin', action='store_true')
    parser.add_argument('-baseline', action='store_true')
    parser.add_argument('-timeParcing', action='store_true')
    parser.parse_args(namespace=sys.modules['__main__'])
 
    lossL = []
    accuracyL = []

    if noAdmin:
        with open('.//test_results//'+fn, 'rb') as f:
            [weightsL, timestamp] = pickle.load(f)
        
        
        (model, x_test, y_test) = build_model()
        print("lodaed model")
        
        for weights in weightsL:            
            model.set_weights(weights)
            loss, accuracy = model.evaluate(x_test, y_test)
            lossL.append(loss)
            accuracyL.append(accuracy)
            print('loss: {}, accuracy: {}'.format(loss, accuracy))
        with open('.//test_results//'+fn+'.log', 'wb') as f2:
            pickle.dump([lossL, accuracyL, timestamp], f2)  
        print("finished evaluating")
#                 
    elif baseline:
#        fn = 'baseline_results_SGD_my_computer.log'

        with open('//home//carmelrab//web_distributed_dnn//baseline//baseline_results.log', 'rb') as f:
            [lossL, accuracyL, timestamp] = pickle.load(f)  

                     

    else:
        with open('//home//carmelrab//web_distributed_dnn//test_results//'+fn, 'rb') as f:
            [lossL, accuracyL, timestamp] = pickle.load(f)
    
    if timeParcing:
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
    plt.stem(timestamp[0:20])
    plt.title('time per epoch')
    plt.ylabel('time [sec]')
    plt.xlabel('epoch')
    fig.savefig('.//test_results//'+fn+'.png')
    plt.show()
    
    send_results_via_mail('.//test_results//'+fn+'.log')
    send_results_via_mail('.//test_results//'+fn+'.png')
