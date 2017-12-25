#! /usr/bin/python

import csv
import matplotlib.pyplot as mpl
import sys
import pickle
import numpy as np
import time
import copy


with open('C://Users//amirli//Desktop//amir//project//baseline_results.log','rb') as f:
    (loss0,acc0,time0) = pickle.load(f)        
with open('C://Users//amirli//Desktop//amir//project//4_clients_if_while_no_ack.pkl.log','rb') as f:
    (loss4,acc4,time4) = pickle.load(f)     
with open('C://Users//amirli//Desktop//amir//project//5_clients_no_ack.pkl.log','rb') as f:
    (loss5,acc5,time5) = pickle.load(f)
with open('C:\\Users\\amirli\\Desktop\\amir\\project\\test_results\\10_clients.log','rb') as f:
    (loss10,acc10,time10) = pickle.load(f)
with open('C:\\Users\\amirli\\Desktop\\amir\\project\\test_results\\8_clients.log','rb') as f:
    (loss8,acc8,time8) = pickle.load(f)
with open('C:\\Users\\amirli\\Desktop\\amir\\project\\test_results\\3_clients.log','rb') as f:
    (loss3,acc3,time3) = pickle.load(f)

        
def calc_avg_time_for_epoch(time_lists):
    avg_time_list = []
    flag_1_client = False #TODO: remove when have 1 client data
    for timeL in time_lists:
        tmp = 0
        for i in range(4,len(timeL)):
            tmp += (timeL[i] - timeL[i-1])
        tmp = tmp/(len(timeL)-4)
        avg_time_list.append(tmp)
    return avg_time_list
    
 
def calc_time_to_70(values_lists,time_lists, acc_lists):
    dic = {}
    for acc_limit in values_lists:
        dic[acc_limit] = []
        for time_elm, acc_elm in zip(time_lists, accL):
            index = [i for i,acc in enumerate(acc_elm) if acc > acc_limit][0]
            dic[acc_limit].append(time_elm[index])
    return dic
            


timeL = [time0, time3, time4, time5, time8, time10]
accL = [acc0, acc3, acc4, acc5, acc8, acc10]
acc_limitL = [0.6, 0.65, 0.7]

# calculate avg time for epoch for each model:
time_per_epoch = calc_avg_time_for_epoch(timeL)


# find time for reaching 70%:
dic_time_to_70 = calc_time_to_70(acc_limitL,timeL,accL)


'''Create the relevant graphs'''

'''Accuracy per time'''
f = mpl.figure()
mainplot = f.add_subplot(111)
mainplot.set_xlabel('time [sec]')
mainplot.set_ylabel('Accuracy [%]')
#plot:
mpl.plot(time0,acc0)  
mpl.plot(time3,acc3)  
mpl.plot(time4,acc4)
mpl.plot(time5,acc5)
mpl.plot(time8,acc8)  
mpl.plot(time10,acc10)
mpl.legend(['baseline', '3 clients','4 clinets', '5 clients', '8 clients', '10 clients'], loc=4)
mpl.title('Accuracy per Time')


'''Accuracy per epoch'''
f3 = mpl.figure()
mainplot = f3.add_subplot(111)
mainplot.set_xlabel('# Epochs')
mainplot.set_ylabel('Accuracy [%]')
#plot:
mpl.plot(acc0)
mpl.plot(acc3)  
mpl.plot(acc4)
mpl.plot(acc5)
mpl.plot(acc8)     
mpl.plot(acc10)
mpl.legend(['baseline', '3 clients','4 clinets', '5 clients', '8 clients', '10 clients'], loc=4)
mpl.title('Accuracy per Epoch')


'''Avrage time for epoch'''
f2 = mpl.figure()
secplot = f2.add_subplot(111)
secplot.set_xlabel('# clients')
secplot.set_ylabel('time [sec]')
secplot.stem(time_per_epoch[1:])
secplot.set_xticks([0,1,2,3,4,5])
legend = ['1', '3', '4', '5', '8', '10']
secplot.set_xticklabels(legend, fontsize=18)
mpl.title('Avarage Time for Epoch')

'''time to reach 70%'''
f3 = mpl.figure()
secplot = f3.add_subplot(111)
secplot.set_xlabel('# clients')
secplot.set_ylabel('time [sec]')
secplot.stem(dic_time_to_70[acc_limitL[-1]][1:])
secplot.set_xticks([0,1,2,3,4,5])
legend = ['3', '4', '5', '8', '10']
secplot.set_xticklabels(legend, fontsize=18)
mpl.title('Time to 70% Accuracy')


mpl.show()

####### SORT 10_clients  file:
#index_list = list(range(71))
#index_list = [x for _,x in sorted(zip(    ,index_list))]
#
#time10 = np.array(time10)[index_list]
#acc10 = np.array(acc10)[index_list]
#loss10 = np.array(loss10)[index_list]   