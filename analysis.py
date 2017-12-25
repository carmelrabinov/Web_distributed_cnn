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

        


####### SORT 10_clients  file:
#index_list = list(range(71))
#index_list = [x for _,x in sorted(zip(time10,index_list))]
#
#time10 = np.array(time10)[index_list]
#acc10 = np.array(acc10)[index_list]
#loss10 = np.array(loss10)[index_list]
#
#
#for i in range(len(time10)-1,-1,-1):
#    for j in range(i-1,-1,-1):
#        if time10[j] > time10[j+1]:
#            tmp_t = time10[j]
#            tmp_a = acc10[j]
#            tmp_l = loss10[j]
#            time10[j] = time10[j+1]
#            loss10[j] = loss10[j+1]
#            acc10[j] = acc10[j+1]
#            time10[j+1] = tmp_t
#            loss10[j+1] = tmp_l
#            acc10[j+1] = tmp_a
#        else:
#            break
#            
#with open('C:\\Users\\amirli\\Desktop\\amir\\project\\test_results\\10_clients_sorted.log','wb') as f:
#    pickle.dump([loss10,acc10,time10],f)
#            
#
#with open('C:\\Users\\amirli\\Desktop\\amir\\project\\test_results\\10_clients_sorted.log','rb') as f:
#    (loss10,acc10,time10) = pickle.load(f)



# save avarage time per epoch:
time = []
legend2 = []

tmp5 = 0
for i in range(4,len(time5)):
    tmp5 += (time5[i] - time5[i-1])
tmp5 = tmp5/(len(time5)-4)


tmp4 = 0
for i in range(4,len(time4)):
    tmp4 += (time4[i] - time4[i-1])
tmp4 = tmp4/(len(time4)-5)

tmp10 = 0
for i in range(4,len(time10)):
    tmp10 += (time10[i] - time10[i-1])
tmp10 = tmp10/(len(time10)-5)

tmp0 = 0
for i in range(4,len(time0)):
    tmp0 += (time0[i] - time0[i-1])
tmp0 = tmp0/(len(time0)-5)

time = [tmp0, 2200, tmp4, tmp5, tmp10]
legend = ['base_line', '1_clients', '4_clients', '5_clients', '10_clients']

dif_time_5 = tmp5-tmp0 
ratio = dif_time_5/tmp5
dif_time_5 = time5[60] - time0[60]
ratio = dif_time_5/time5[60]

if True:
    '''Create the relevant graph'''
    #create the plot, axeses and titles:
    f = mpl.figure()
    mainplot = f.add_subplot(111)
    mainplot.yaxis.tick_right()
    mainplot.set_xlabel('# time')
    mainplot.set_ylabel('# Accuracy')
    mainplot.yaxis.set_label_position("right")
    
    
    #plot:
    mpl.plot(time4,acc4,'b-')
    mpl.plot(time5,acc5,'g-')
    mpl.plot(time10,acc10,'m-')
    mpl.plot(time0,acc0,'r-')
    

    mpl.legend(['4 clinets', '5 clients', '10 clients', 'baseline'], loc=4)
    
    f3 = mpl.figure()
    mainplot = f3.add_subplot(111)
    mainplot.yaxis.tick_right()
    mainplot.set_xlabel('# Epoch')
    mainplot.set_ylabel('# Accuracy')
    mainplot.yaxis.set_label_position("right")
    

    #plot:
    mpl.plot(acc4,'b-')
    mpl.plot(acc5,'g-')
    mpl.plot(acc10,'m-')
    mpl.plot(acc0,'r-')
    
    mpl.legend(['4 clinets', '5 clients', '10 clients', 'baseline'], loc=4)
    
    f2 = mpl.figure()

    secplot = f2.add_subplot(111)
#    secplot.yaxis.tick_right()
    secplot.set_xlabel('# clients')
    secplot.set_ylabel('# time per epoch')
#    secplot.yaxis.set_label_position("right")
    secplot.stem(time)
    secplot.set_xticks([0,1,2,3,4])
    secplot.set_xticklabels(legend, rotation='vertical', fontsize=18)


    mpl.show()

    