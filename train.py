import lightgbm as lgb
import numpy as np
from sklearn.model_selection import KFold
from param import *

train = np.loadtxt('features/train_data_16d.txt')
y = np.loadtxt('features/label_16d.txt')
userId = np.loadtxt('features/userId_16d.txt',dtype=str)
test = np.loadtxt('features/test_data_16d.txt')
test_y = np.zeros(y.shape)
pred_y = np.zeros((test.shape[0],))

kf = KFold(n_splits=5, shuffle=True, random_state=1024)

for train_index,val_index in kf.split(train):
    X_train,X_val = train[train_index],train[val_index]
    y_train,y_val = y[train_index],y[val_index]
    clf = lgb.LGBMClassifier(learning_rate=0.02,objective='binary',reg_alpha=0.002,
                             subsample=0.8,colsample_bytree=0.8,n_estimators=50000,
                             early_stopping_round=300,silent=-1)
    clf.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_val,y_val)],
            eval_metric={'binary_logloss'},verbose=100,early_stopping_rounds=300)
    test_y[val_index] = clf.predict_proba(X_val,num_iteration=clf.best_iteration_)[:,1]
    pred_y += clf.predict_proba(test,num_iteration=clf.best_iteration_)[:,1]

res = {}
for i in range(len(pred_y)):
    res[userId[i]] = pred_y[i]
D = sorted(res.items(),key=lambda x:x[1],reverse=True)
with open('submission.txt','w') as f:
    for i in range(THRESHOLD):
        f.write(D[i][0]+'\n')
