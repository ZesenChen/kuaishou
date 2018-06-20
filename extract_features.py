from param import *
import numpy as np
import pandas as pd
from func import *

train_register = {}
test_register = {}
train_dev_type = {}
test_dev_type = {}
train_reg_type = {}
test_reg_type = {}


ur = pd.read_csv('user_register_log.txt',sep='\t',header=None)
D=dict(ur[3].value_counts())
#ur['dev_type'] = ur[3].apply(lambda x:D[x])
#print('dev type is ready!')
count = 0
with open('user_register_log.txt','r') as f:
    while True:
        count += 1
        if count % 500 == 0:
            print(count)
        line = f.readline()
        if not line:
            break
        tmp = line.strip().split()
        if int(tmp[1]) in TRAIN_REGISTER_DAT:
            train_register[tmp[0]] = int(tmp[1])
            train_dev_type[tmp[0]] = int(tmp[3])
            train_reg_type[tmp[0]] = np.zeros((12,))
            train_reg_type[tmp[0]][int(tmp[2])] = 1
        if int(tmp[1]) in TEST_REGISTER_DAT:
            test_register[tmp[0]] = int(tmp[1])
            test_dev_type[tmp[0]] = int(tmp[3])
            test_reg_type[tmp[0]] = np.zeros((12,))
            test_reg_type[tmp[0]][int(tmp[2])] = 1

for Id in train_register:
    train_dev_type[Id] = D[train_dev_type[Id]] 
for Id in test_register:
    test_dev_type[Id] = D[test_dev_type[Id]]


print('register is ready!')
print('train data len is: '+str(len(train_register)))
print('test data len is: '+str(len(test_register)))

train_launch = {}
test_launch = {}
max_min_train_launch = {}
max_min_test_launch = {}

labels = {}
for Id in train_register:
    train_launch[Id] = np.zeros((len(TRAIN_ACT_DAT),),dtype=np.int)
    max_min_train_launch[Id] = np.array([0,32])
    labels[Id] = 0

for Id in test_register:
    max_min_test_launch[Id] = np.array([0,32])
    test_launch[Id] = np.zeros((len(TRAIN_ACT_DAT),),dtype=np.int)

#labels = dict.fromkeys(train_register.keys(),0)

with open('app_launch_log.txt','r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        tmp = line.strip().split()
        if tmp[0] in train_register and int(tmp[1]) in TRAIN_PREDICT_DAY:
            labels[tmp[0]] = 1
        if tmp[0] in train_register and int(tmp[1]) in TRAIN_ACT_DAT:
            max_min_train_launch[tmp[0]][0] = max(max_min_train_launch[tmp[0]][0],
                                                  TRAIN_PREDICT_DAY[0]-int(tmp[1]))
            max_min_train_launch[tmp[0]][1] = min(max_min_train_launch[tmp[0]][1],
                                                  TRAIN_PREDICT_DAY[0]-int(tmp[1]))
            train_launch[tmp[0]][int(tmp[1])-TRAIN_ACT_DAT[0]] += 1
        if tmp[0] in test_register and int(tmp[1]) in TEST_ACT_DAT:
            max_min_test_launch[tmp[0]][0] = max(max_min_test_launch[tmp[0]][0],
                                                  TEST_PREDICT_DAY[0]-int(tmp[1]))
            max_min_test_launch[tmp[0]][1] = min(max_min_test_launch[tmp[0]][1],
                                                  TEST_PREDICT_DAY[0]-int(tmp[1]))
            test_launch[tmp[0]][int(tmp[1])-TEST_ACT_DAT[0]] += 1

#print(train_launch)

'''
for Id in train_launch:
    train_launch[Id] = train_launch[Id]*WEIGHT
for Id in test_launch:
    test_launch[Id] = test_launch[Id]*WEIGHT
'''
train_video = {}#dict.fromkeys(train_register.keys(),
                             #np.zeros((len(TRAIN_ACT_DAT),)))
test_video = {}#dict.fromkeys(test_register.keys(),
                            #np.zeros((len(TRAIN_ACT_DAT),)))
max_min_train_video = {}
max_min_test_video = {}

launch_video_train = {}
launch_video_test = {}

for Id in train_register:
    max_min_train_video[Id] = np.array([0,32])
    train_video[Id] = np.zeros((len(TRAIN_ACT_DAT),),dtype=np.int)
    #launch_video_train[Id] = np.zeros((len(TRAIN_ACT_DAT),),dtype=np.int)

for Id in test_register:
    max_min_test_video[Id] = np.array([0,32])
    test_video[Id] = np.zeros((len(TRAIN_ACT_DAT),),dtype=np.int)
    #launch_video_test[Id] = np.zeros((len(TRAIN_ACT_DAT),),dtype=np.int)

with open('video_create_log.txt','r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        tmp = line.strip().split()
        if tmp[0] in train_register and int(tmp[1]) in TRAIN_PREDICT_DAY:
            labels[tmp[0]] = 1
        if tmp[0] in train_register and int(tmp[1]) in TRAIN_ACT_DAT:
            max_min_train_video[tmp[0]][0] = max(max_min_train_video[tmp[0]][0],
                                                  TRAIN_PREDICT_DAY[0]-int(tmp[1]))
            max_min_train_video[tmp[0]][1] = min(max_min_train_video[tmp[0]][1],
                                                  TRAIN_PREDICT_DAY[0]-int(tmp[1]))
            train_video[tmp[0]][int(tmp[1])-TRAIN_ACT_DAT[0]] += 1
        if tmp[0] in test_register and int(tmp[1]) in TEST_ACT_DAT:
            max_min_test_video[tmp[0]][0] = max(max_min_test_video[tmp[0]][0],
                                                  TEST_PREDICT_DAY[0]-int(tmp[1]))
            max_min_test_video[tmp[0]][1] = min(max_min_test_video[tmp[0]][1],
                                                  TEST_PREDICT_DAY[0]-int(tmp[1]))
            test_video[tmp[0]][int(tmp[1])-TEST_ACT_DAT[0]] += 1
        
print('video is ready!')
'''
for Id in train_video:
    train_video[Id] = train_video[Id]*WEIGHT
for Id in test_video:
    test_video[Id] = test_video[Id]*WEIGHT
'''
train_act_day = {}#dict.fromkeys(train_register.keys(),
                              #np.zeros((len(TRAIN_ACT_DAT),)))
test_act_day = {}#dict.fromkeys(test_register.keys(),
                             #np.zeros((len(TRAIN_ACT_DAT),)))
train_page = {}#dict.fromkeys(train_register.keys(),
                              #np.zeros((5,)))
test_page = {}#dict.fromkeys(test_register.keys(),
                              #np.zeros((5,)))
train_act_type = {}#dict.fromkeys(train_register.keys(),
                               #np.zeros((6,)))
test_act_type = {}#dict.fromkeys(test_register.keys(),

train_video_id = {}

test_video_id = {}                              #np.zeros((6,)))

train_author_id = {}

test_author_id = {}

for Id in train_register:
    train_act_day[Id] = np.zeros((len(TRAIN_ACT_DAT),),dtype=np.int)
    train_page[Id] = np.zeros((5,),dtype=np.int)
    train_act_type[Id] = np.zeros((6,),dtype=np.int)
    launch_video_train[Id] = train_video[Id]*train_launch[Id]
    train_video_id[Id] = []
    train_author_id[Id] = []

for Id in test_register:
    test_act_day[Id] = np.zeros((len(TRAIN_ACT_DAT),),dtype=np.int)
    test_page[Id] = np.zeros((5,),dtype=np.int)
    test_act_type[Id] = np.zeros((6,),dtype=np.int)
    launch_video_test[Id] = test_video[Id]*test_launch[Id]
    test_video_id[Id] = []
    test_author_id[Id] = []    

count = 0
with open('user_activity_log.txt','r') as f:
    while True:
        line = f.readline()
        count += 1
        if count%100000==0:
            print(count)
        if not line:
            break
        tmp = line.strip().split()
        if tmp[0] in train_register and int(tmp[1]) in TRAIN_PREDICT_DAY:
            labels[tmp[0]] = 1
        if tmp[0] in train_register and int(tmp[1]) in TRAIN_ACT_DAT:
            train_act_day[tmp[0]][int(tmp[1])-TRAIN_ACT_DAT[0]] += 1
            train_page[tmp[0]][int(tmp[2])] += 1
            train_act_type[tmp[0]][int(tmp[-1])] += 1
        if tmp[0] in test_register and int(tmp[1]) in TEST_ACT_DAT:
            test_act_day[tmp[0]][int(tmp[1])-TEST_ACT_DAT[0]] += 1
            test_page[tmp[0]][int(tmp[2])] += 1
            test_act_type[tmp[0]][int(tmp[-1])] += 1
           
print('activity is ready!')
#f = open('train/train_data.txt','w')
#l = open('train/labels.txt','w')
'''
for Id in train_act_day:
    train_act_day[Id] = train_act_day[Id]*WEIGHT
for Id in test_act_day:
    test_act_day[Id] = test_act_day[Id]*WEIGHT
'''
count = 0
y = []
train_data = []
for Id in train_register:
    print(Id)
    y.append(labels[Id])
    tmp = np.hstack((np.array([TRAIN_PREDICT_DAY[0]-train_register[Id]]),
                     train_reg_type[Id],
                     np.array([train_dev_type[Id]]),
                     np.array([sum(train_launch[Id]),np.mean(train_launch[Id]),np.std(train_launch[Id]),conseq(train_launch[Id])]),
                     np.array([sum(train_launch[Id]*WEIGHT),np.mean(train_launch[Id]*WEIGHT),np.std(train_launch[Id]*WEIGHT)]),
                     train_launch[Id],
                     max_min_train_launch[Id],
                     np.array([sum(train_video[Id]),np.mean(train_video[Id]),np.std(train_video[Id]),np.max(train_video[Id]),np.min(train_video[Id]),conseq(train_video[Id])]),
                     max_min_train_video[Id],
                     np.array([sum(train_video[Id]*WEIGHT),np.mean(train_video[Id]*WEIGHT),np.std(train_video[Id]*WEIGHT),np.max(train_video[Id]*WEIGHT),np.min(train_video[Id]*WEIGHT)]),
                     train_video[Id],
                     np.array([sum(launch_video_train[Id]),np.mean(launch_video_train[Id]),np.std(launch_video_train[Id]),np.max(launch_video_train[Id]),np.min(launch_video_train[Id]),conseq(launch_video_train[Id])]),
                     np.array([sum(launch_video_train[Id]*WEIGHT),np.mean(launch_video_train[Id]*WEIGHT),np.std(launch_video_train[Id]*WEIGHT),np.max(launch_video_train[Id]*WEIGHT),np.min(launch_video_train[Id]*WEIGHT)]),
                     launch_video_train[Id],
                     np.array([sum(train_act_day[Id]),np.mean(train_act_day[Id]),np.std(train_act_day[Id]),np.max(train_act_day[Id]),np.min(train_act_day[Id]),conseq(train_act_day[Id])]),
                     np.array([sum(train_act_day[Id]*WEIGHT),np.mean(train_act_day[Id]*WEIGHT),np.std(train_act_day[Id]*WEIGHT),np.max(train_act_day[Id]*WEIGHT),np.min(train_act_day[Id]*WEIGHT)]),
                     train_act_day[Id],
                     np.array([sum(train_page[Id]),np.mean(train_page[Id]),np.std(train_page[Id]),np.max(train_page[Id]),np.min(train_page[Id])]),
                     train_page[Id],
                     np.array([sum(train_page[Id]!=0)]),
                     np.array([sum(train_act_type[Id]),np.mean(train_act_type[Id]),np.std(train_act_type[Id]),np.max(train_act_type[Id]),np.min(train_act_type[Id])]),
                     train_act_type[Id],
                     np.array([sum(train_act_type[Id]!=0)])
                     ))
    train_data.append(tmp)
np.savetxt('features/labels_2w_620_2.txt',np.array(y))
np.savetxt('features/train_data_2w_620_2.txt',np.array(train_data))

f = open('features/userId_2w_620.txt','w')
print('train data is ready!')
test_data = []
for Id in test_register:
    print(Id)
    f.write(Id+'\n')
    tmp = np.hstack((np.array([TEST_PREDICT_DAY[0]-test_register[Id]]),
                     test_reg_type[Id],
                     np.array([test_dev_type[Id]]),
                     np.array([sum(test_launch[Id]),np.mean(test_launch[Id]),np.std(test_launch[Id]),conseq(test_launch[Id])]),
                     np.array([sum(test_launch[Id]*WEIGHT),np.mean(test_launch[Id]*WEIGHT),np.std(test_launch[Id]*WEIGHT)]),
                     test_launch[Id],
                     max_min_test_launch[Id],
                     np.array([sum(test_video[Id]),np.mean(test_video[Id]),np.std(test_video[Id]),np.max(test_video[Id]),np.min(test_video[Id]),conseq(test_video[Id])]),
                     max_min_test_video[Id],
                     np.array([sum(test_video[Id]*WEIGHT),np.mean(test_video[Id]*WEIGHT),np.std(test_video[Id]*WEIGHT),np.max(test_video[Id]*WEIGHT),np.min(test_video[Id]*WEIGHT)]),
                     test_video[Id],
                     np.array([sum(launch_video_test[Id]),np.mean(launch_video_test[Id]),np.std(launch_video_test[Id]),np.max(launch_video_test[Id]),np.min(launch_video_test[Id]),conseq(launch_video_test[Id])]),
                     np.array([sum(launch_video_test[Id]*WEIGHT),np.mean(launch_video_test[Id]*WEIGHT),np.std(launch_video_test[Id]*WEIGHT),np.max(launch_video_test[Id]*WEIGHT),np.min(launch_video_test[Id]*WEIGHT)]),
                     launch_video_test[Id],
                     np.array([sum(test_act_day[Id]),np.mean(test_act_day[Id]),np.std(test_act_day[Id]),np.max(test_act_day[Id]),np.min(test_act_day[Id]),conseq(test_act_day[Id])]),
                     np.array([sum(test_act_day[Id]*WEIGHT),np.mean(test_act_day[Id]*WEIGHT),np.std(test_act_day[Id]*WEIGHT),np.max(test_act_day[Id]*WEIGHT),np.min(test_act_day[Id]*WEIGHT)]),
                     test_act_day[Id],
                     np.array([sum(test_page[Id]),np.mean(test_page[Id]),np.std(test_page[Id]),np.max(test_page[Id]),np.min(test_page[Id])]),
                     test_page[Id],
                     np.array([sum(test_page[Id]!=0)]),
                     np.array([sum(test_act_type[Id]),np.mean(test_act_type[Id]),np.std(test_act_type[Id]),np.max(test_act_type[Id]),np.min(test_act_type[Id])]),
                     test_act_type[Id],
                     np.array([sum(test_act_type[Id]!=0)])
                     ))   
    test_data.append(tmp) 
print('all is ready!')
f.close()
np.savetxt('features/test_data_2w_620.txt',np.array(test_data))
