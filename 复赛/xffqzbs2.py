import os
# os.system('pip install catboost==0.15.2')
# os.system('pip install tqdm')
# os.system('pip install xlrd')


import numpy as np
import pandas as pd
from pandas import DataFrame as DF
import scipy.spatial.distance as dist
import catboost as cbt
import json
from sklearn.metrics import f1_score
import time
import catboost as cbt
import gc
import math
from tqdm import tqdm
from scipy import stats
from sklearn.cluster import KMeans
from six.moves import reduce
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import matplotlib.pyplot as plt
from datetime import datetime,timedelta


import os
# os.chdir('/cos_person/xf2019/')
path = '/data1/pengjiaxin/'


train1 = pd.read_table(path+"round1_iflyad_anticheat_traindata.txt")
print(len(train1))
train2 = pd.read_table(path+"round2_iflyad_anticheat_traindata.txt")
print(len(train2))

print(len(train1)+len(train2))
train = pd.concat([train1, train2])
del train1, train2
train = train.drop_duplicates()
print(len(train))

testA = pd.read_table(path+"round2_iflyad_anticheat_testdata_feature_A.txt")
testB = pd.read_table(path+"round2_iflyad_anticheat_testdata_feature_B.txt")


test = pd.concat([testA, testB])
data = train.append(test).reset_index(drop=True)
city_province = pd.read_excel('/data1/pengjiaxin/city_province.xlsx')
data = data.merge(city_province, on = 'city', how = 'left')

data['time'] = pd.to_datetime(data['nginxtime']*1e+6) + timedelta(hours=8)
data['day'] = data['time'].dt.day
data['hour'] = data['time'].dt.hour

data['size'] = (np.sqrt(data['h']**2 + data['w'] ** 2) / 2.54) / 1000
data['ratio'] = data['h'] / data['w']
data['px'] = data['ppi'] * data['size']
data['mj'] = data['h'] * data['w']

data['ip_0'] = data['ip'].map(lambda x:'.'.join(x.split('.')[:1]))
data['ip_1'] = data['ip'].map(lambda x:'.'.join(x.split('.')[0:2]))
data['ip_2'] = data['ip'].map(lambda x:'.'.join(x.split('.')[0:3]))

data['reqrealip_0'] = data['reqrealip'].map(lambda x:'.'.join(str(x).split('.')[:1]))
data['reqrealip_1'] = data['reqrealip'].map(lambda x:'.'.join(str(x).split('.')[0:2]))
data['reqrealip_2'] = data['reqrealip'].map(lambda x:'.'.join(str(x).split('.')[0:3]))

data['ip_equal'] = (data['ip'] == data['reqrealip']).astype(int)

ip_feat = ['ip_0','ip_1','ip_2','reqrealip_0','reqrealip_1','reqrealip_2','ip_equal']

object_col = [i for i in data.select_dtypes(object).columns if i not in ['sid','label']]
for i in tqdm(object_col):
    lbl = LabelEncoder()
    data[i] = lbl.fit_transform(data[i].astype(str))
    
cat_list = [i for i in train.columns if i not in ['sid','label','nginxtime']] + ['hour', 'province_new'] + ip_feat
for i in tqdm(cat_list):
    data['{}_count'.format(i)] = data.groupby(['{}'.format(i)])['sid'].transform('count')
    
feature_name = [i for i in data.columns if i not in ['sid', 'label','time','day']]
gc.collect()
req_col = ['reqrealip_0','reqrealip_1','reqrealip_2']
data[req_col].astype(float)

print()
print(set(data.columns))
col = ['day', 'reqrealip_2_count', 'ntt', 'reqrealip_count', 'orientation_count', 'imeimd5_count', 'city', 'dvctype',
       'px', 'ip_2', 'openudidmd5', 'reqrealip', 'h', 'adunitshowid_count', 'hour', 'orientation', 'carrier',
       'apptype_count', 'province_new', 'city_count', 'mediashowid', 'hour_count', 'ip_0', 'ip_count', 'ip_equal_count',
       'lan', 'reqrealip_2', 'os', 'province', 'ratio', 'lan_count', 'ip_equal', 'os_count', 'reqrealip_1', 'ip_1',
       'nginxtime', 'make', 'ppi_count', 'ntt_count', 'openudidmd5_count', 'make_count', 'imeimd5', 'pkgname', 'macmd5',
       'ppi', 'w', 'ver_count', 'ip_0_count', 'adidmd5_count', 'reqrealip_0', 'reqrealip_0_count', 'ip_1_count', 'ver',
       'ip', 'province_new_count', 'model', 'osv_count', 'carrier_count', 'ip_2_count', 'province_count', 'time',
       'dvctype_count', 'mj', 'apptype', 'size', 'reqrealip_1_count', 'adidmd5', 'idfamd5', 'w_count', 'adunitshowid',
       'idfamd5_count', 'macmd5_count', 'h_count', 'pkgname_count', 'mediashowid_count', 'model_count', 'sid', 'osv']

data[col] = data[col].fillna(9999)

tr_index = ~data['label'].isnull()
X_train = data[tr_index][list(set(feature_name))].reset_index(drop=True)
y = data[tr_index]['label'].reset_index(drop=True).astype(int)
X_test = data[~tr_index][list(set(feature_name))].reset_index(drop=True)
print(X_train.shape, X_test.shape)
oof = np.zeros(X_train.shape[0])
prediction = np.zeros(X_test.shape[0])
seeds = [19970412, 2019 * 2 + 1024, 4096, 2048, 1024]
num_model_seed = 1
for model_seed in range(num_model_seed):
    oof_cat = np.zeros(X_train.shape[0])
    prediction_cat=np.zeros(X_test.shape[0])
    skf = StratifiedKFold(n_splits=5, random_state=seeds[model_seed], shuffle=True)
    for index, (train_index, test_index) in enumerate(skf.split(X_train, y)):
        print(index)
        train_x, test_x, train_y, test_y = X_train[feature_name].iloc[train_index], X_train[feature_name].iloc[test_index], \
                                           y.iloc[train_index], y.iloc[test_index]
        cbt_model = cbt.CatBoostClassifier(iterations=5000, learning_rate=0.1,max_depth=7,verbose=100,
                                       early_stopping_rounds=500, task_type='GPU',eval_metric='F1',
                                       cat_features=cat_list)
        cbt_model.fit(train_x[feature_name], train_y, eval_set=(test_x[feature_name],test_y))
        gc.collect()    
        oof_cat[test_index] += cbt_model.predict_proba(test_x)[:, 1]
        prediction_cat += cbt_model.predict_proba(X_test[feature_name])[:,1]/5
    print('F1',f1_score(y, np.round(oof_cat)))    
    oof += oof_cat / num_model_seed
    prediction += prediction_cat / num_model_seed
print('score', f1_score(y, np.round(oof)))
# write to csv
submit = test[['sid']]
submit['label'] = (prediction>=0.499).astype(int)
print(submit['label'].value_counts())
submit.to_csv("/data1/pengjiaxin/round2/submission0823.csv", index=False)

