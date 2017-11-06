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
import numpy as np



optimizer = 'SGD'
with open('C:\\Users\carmelr\\projectA\\model_info'+optimizer+'.log', 'rb') as f:
    [acc, val_acc, loss, val_loss, time_] = pickle.load(f)
   
# summarize history for loss
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(loss)
plt.plot(val_loss) 
plt.title('model loss')
plt.ylabel('loss')
plt.legend(['train', 'validation'], loc='upper left')

plt.subplot(2, 1, 2)
plt.plot(acc)
plt.plot(val_acc) 
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.show()
#    plt.savefig('temp.png')
fig.savefig('C:\\Users\carmelr\\projectA\\model_info'+optimizer+'.png')
