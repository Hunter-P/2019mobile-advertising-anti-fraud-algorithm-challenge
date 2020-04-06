# -*- coding: utf-8 -*-
# @Time    : 2019/8/18 9:28
# @Author  : YYLin
# @Email   : 854280599@qq.com
# @File    : A_Simple_Stacking_Model.py

# 特征部分选择使用之前简单的特征 加上lgb catboost xgb进行stacking操作 分数大约46
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta
import catboost as cbt
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')
from scipy import stats

# 读取训练集中的数据
use_train = True
if use_train:
    test = pd.read_table("../A_Data/testdata.txt", usecols=['sid'])
    all_data = pd.read_csv("../A_Data/new_data_keep_all_attr.csv", sep=',')
else:
    test = pd.read_table("../A_Data/testdata.txt", nrows=1000)
    train = pd.read_table("../A_Data/traindata.txt", nrows=1000)
    all_data = pd.concat([train, test], ignore_index=True)

all_data['time'] = pd.to_datetime(all_data['nginxtime'] * 1e+6) + timedelta(hours=8)
all_data['day'] = all_data['time'].dt.dayofyear
all_data['hour'] = all_data['time'].dt.hour

all_data['model'].replace('PACM00', "OPPO R15", inplace=True)
all_data['model'].replace('PBAM00', "OPPO A5", inplace=True)
all_data['model'].replace('PBEM00', "OPPO R17", inplace=True)
all_data['model'].replace('PADM00', "OPPO A3", inplace=True)
all_data['model'].replace('PBBM00', "OPPO A7", inplace=True)
all_data['model'].replace('PAAM00', "OPPO R15_1", inplace=True)
all_data['model'].replace('PACT00', "OPPO R15_2", inplace=True)
all_data['model'].replace('PABT00', "OPPO A5_1", inplace=True)
all_data['model'].replace('PBCM10', "OPPO R15x", inplace=True)

for fea in ['model', 'make', 'lan', 'new_make', 'new_model']:
    all_data[fea] = all_data[fea].astype('str')
    all_data[fea] = all_data[fea].map(lambda x: x.upper())

    from urllib.parse import unquote


    def url_clean(x):
        x = unquote(x, 'utf-8').replace('%2B', ' ').replace('%20', ' ').replace('%2F', '/').replace('%3F', '?').replace(
            '%25', '%').replace('%23', '#').replace(".", ' ').replace('??', ' '). \
            replace('%26', ' ').replace("%3D", '=').replace('%22', '').replace('_', ' ').replace('+', ' ').replace('-',
                                                                                                                   ' ').replace(
            '__', ' ').replace('  ', ' ').replace(',', ' ')

        if (x[0] == 'V') & (x[-1] == 'A'):
            return "VIVO {}".format(x)
        elif (x[0] == 'P') & (x[-1] == '0'):
            return "OPPO {}".format(x)
        elif (len(x) == 5) & (x[0] == 'O'):
            return "Smartisan {}".format(x)
        elif ('AL00' in x):
            return "HW {}".format(x)
        else:
            return x


    all_data[fea] = all_data[fea].map(url_clean)

all_data['big_model'] = all_data['model'].map(lambda x: x.split(' ')[0])
all_data['model_equal_make'] = (all_data['big_model'] == all_data['make']).astype(int)

# 处理 ntt 的数据特征 但是不删除之前的特征 将其归为新的一列数据
all_data['new_ntt'] = all_data['ntt']
all_data.new_ntt[(all_data.new_ntt == 0) | (all_data.new_ntt == 7)] = 0
all_data.new_ntt[(all_data.new_ntt == 1) | (all_data.new_ntt == 2)] = 1
all_data.new_ntt[all_data.new_ntt == 3] = 2
all_data.new_ntt[(all_data.new_ntt >= 4) & (all_data.new_ntt <= 6)] = 3

# 使用make填充 h w ppi值为0.0的数据
all_data['h'].replace(0.0, np.nan, inplace=True)
all_data['w'].replace(0.0, np.nan, inplace=True)
# all_data['ppi'].replace(0.0, np.nan, inplace=True)
# cols = ['h', 'w', 'ppi']
cols = ['h', 'w']
gp_col = 'make'
for col in tqdm(cols):
    na_series = all_data[col].isna()
    names = list(all_data.loc[na_series, gp_col])
    # 使用均值 或者众数进行填充缺失值
    # df_fill = all_data.groupby(gp_col)[col].mean()
    df_fill = all_data.groupby(gp_col)[col].agg(lambda x: stats.mode(x)[0][0])
    t = df_fill.loc[names]
    t.index = all_data.loc[na_series, col].index
    # 相同的index进行赋值
    all_data.loc[na_series, col] = t
    all_data[col].fillna(0.0, inplace=True)
    del df_fill
    gc.collect()

# H, W, PPI
all_data['size'] = (np.sqrt(all_data['h'] ** 2 + all_data['w'] ** 2) / 2.54) / 1000
all_data['ratio'] = all_data['h'] / all_data['w']
all_data['px'] = all_data['ppi'] * all_data['size']
all_data['mj'] = all_data['h'] * all_data['w']

# 强特征进行组合
Fusion_attributes = ['make_adunitshowid', 'adunitshowid_model', 'adunitshowid_ratio', 'make_model',
                     'make_osv', 'make_ratio', 'model_osv', 'model_ratio', 'model_h', 'ratio_osv']

for attribute in tqdm(Fusion_attributes):
    name = "Fusion_attr_" + attribute
    dummy = 'label'
    cols = attribute.split("_")
    cols_with_dummy = cols.copy()
    cols_with_dummy.append(dummy)
    gp = all_data[cols_with_dummy].groupby(by=cols)[[dummy]].count().reset_index().rename(index=str,
                                                                                          columns={dummy: name})
    all_data = all_data.merge(gp, on=cols, how='left')

# 对ip地址和reqrealip地址进行分割 定义一个machine的关键字
all_data['ip2'] = all_data['ip'].apply(lambda x: '.'.join(x.split('.')[0:2]))
all_data['ip3'] = all_data['ip'].apply(lambda x: '.'.join(x.split('.')[0:3]))
all_data['reqrealip2'] = all_data['reqrealip'].apply(lambda x: '.'.join(x.split('.')[0:2]))
all_data['reqrealip3'] = all_data['reqrealip'].apply(lambda x: '.'.join(x.split('.')[0:3]))
all_data['machine'] = 1000 * all_data['model'] + all_data['make']

var_mean_attributes = ['adunitshowid', 'make', 'model', 'ver']
for attr in tqdm(var_mean_attributes):
    # 统计关于ratio的方差和均值特征
    var_label = 'ratio'
    var_name = 'var_' + attr + '_' + var_label
    gp = all_data[[attr, var_label]].groupby(attr)[var_label].var().reset_index().rename(index=str,
                                                                                         columns={var_label: var_name})
    all_data = all_data.merge(gp, on=attr, how='left')
    all_data[var_name] = all_data[var_name].fillna(0).astype(int)

    mean_label = 'ratio'
    mean_name = 'mean_' + attr + '_' + mean_label
    gp = all_data[[attr, mean_label]].groupby(attr)[mean_label].mean().reset_index().rename(index=str, columns={
        mean_label: mean_name})
    all_data = all_data.merge(gp, on=attr, how='left')
    all_data[mean_name] = all_data[mean_name].fillna(0).astype(int)

    # 统计关于h的方差和均值特征
    var_label = 'h'
    var_name = 'var_' + attr + '_' + var_label
    gp = all_data[[attr, var_label]].groupby(attr)[var_label].var().reset_index().rename(index=str,
                                                                                         columns={var_label: var_name})
    all_data = all_data.merge(gp, on=attr, how='left')
    all_data[var_name] = all_data[var_name].fillna(0).astype(int)

    mean_label = 'h'
    mean_name = 'mean_' + attr + '_' + mean_label
    gp = all_data[[attr, mean_label]].groupby(attr)[mean_label].mean().reset_index().rename(index=str, columns={
        mean_label: mean_name})
    all_data = all_data.merge(gp, on=attr, how='left')
    all_data[mean_name] = all_data[mean_name].fillna(0).astype(int)

    # 统计关于h的方差和均值特征
    var_label = 'w'
    var_name = 'var_' + attr + '_' + var_label
    gp = all_data[[attr, var_label]].groupby(attr)[var_label].var().reset_index().rename(index=str,
                                                                                         columns={var_label: var_name})
    all_data = all_data.merge(gp, on=attr, how='left')
    all_data[var_name] = all_data[var_name].fillna(0).astype(int)

    mean_label = 'w'
    mean_name = 'mean_' + attr + '_' + mean_label
    gp = all_data[[attr, mean_label]].groupby(attr)[mean_label].mean().reset_index().rename(index=str, columns={
        mean_label: mean_name})
    all_data = all_data.merge(gp, on=attr, how='left')
    all_data[mean_name] = all_data[mean_name].fillna(0).astype(int)

    del gp
    gc.collect()

cat_col = [i for i in all_data.select_dtypes(object).columns if i not in ['sid', 'label']]
for i in tqdm(cat_col):
    lbl = LabelEncoder()
    all_data['count_' + i] = all_data.groupby([i])[i].transform('count')
    all_data[i] = lbl.fit_transform(all_data[i].astype(str))

for i in tqdm(['h', 'w', 'ppi', 'ratio']):
    all_data['{}_count'.format(i)] = all_data.groupby(['{}'.format(i)])['sid'].transform('count')

feature_name = [i for i in all_data.columns if i not in ['sid', 'label', 'time']]
print(feature_name)
print('all_data.info:', all_data.info())

cat_list = ['pkgname', 'ver', 'adunitshowid', 'mediashowid', 'apptype', 'ip', 'city', 'province', 'reqrealip',
            'adidmd5',
            'imeimd5', 'idfamd5', 'openudidmd5', 'macmd5', 'dvctype', 'model', 'make', 'ntt', 'carrier', 'os', 'osv',
            'orientation', 'lan', 'h', 'w', 'ppi', 'ip2', 'new_make', 'new_model', 'country', 'new_province',
            'new_city',
            'ip3', 'reqrealip2', 'reqrealip3']

tr_index = ~all_data['label'].isnull()
X_train = all_data[tr_index][list(set(feature_name))].reset_index(drop=True)
y = all_data[tr_index]['label'].reset_index(drop=True).astype(int)
X_test = all_data[~tr_index][list(set(feature_name))].reset_index(drop=True)
print(X_train.shape, X_test.shape)
# 节约一下内存
del all_data
gc.collect()


# 以下代码是5折交叉验证的结果 + lgb catboost xgb 最后使用logist进行回归预测
def get_stacking(clf, x_train, y_train, x_test, feature_name, n_folds=5):
    print('len_x_train:', len(x_train))

    train_num, test_num = x_train.shape[0], x_test.shape[0]
    second_level_train_set = np.zeros((train_num,))
    second_level_test_set = np.zeros((test_num,))
    test_nfolds_sets = np.zeros((test_num, n_folds))
    kf = KFold(n_splits=n_folds)

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tra, y_tra = x_train[feature_name].iloc[train_index], y_train[train_index]
        x_tst, y_tst = x_train[feature_name].iloc[test_index], y_train[test_index]

        clf.fit(x_tra[feature_name], y_tra, eval_set=(x_tst[feature_name], y_tst))

        second_level_train_set[test_index] = clf.predict(x_tst[feature_name])
        test_nfolds_sets[:, i] = clf.predict(x_test[feature_name])

    second_level_test_set[:] = test_nfolds_sets.mean(axis=1)
    return second_level_train_set, second_level_test_set


def lgb_f1(labels, preds):
    score = f1_score(labels, np.round(preds))
    return 'f1', score, True


lgb_model = lgb.LGBMClassifier(random_seed=2019, n_jobs=-1, objective='binary', learning_rate=0.05, n_estimators=3000,
                               num_leaves=31, max_depth=-1, min_child_samples=50, min_child_weight=9, subsample_freq=1,
                               subsample=0.7, colsample_bytree=0.7, reg_alpha=1, reg_lambda=5, eval_metric=lgb_f1,
                               early_stopping_rounds=400)

xgb_model = xgb.XGBRegressor(max_depth=5, learning_rate=0.05, n_estimators=3000, silent=False, objective='binary',
                             early_stopping_rounds=400, feval=lgb_f1)

cbt_model = cbt.CatBoostClassifier(iterations=3000, learning_rate=0.05, max_depth=11, l2_leaf_reg=1, verbose=10,
                                   early_stopping_rounds=400, task_type='GPU', eval_metric='F1', cat_features=cat_list)

train_sets = []
test_sets = []
for clf in [cbt_model, lgb_model, xgb_model]:
    print('begin train clf:', clf)
    train_set, test_set = get_stacking(clf, X_train, y, X_test, feature_name)
    train_sets.append(train_set)
    test_sets.append(test_set)

meta_train = np.concatenate([result_set.reshape(-1, 1) for result_set in train_sets], axis=1)
meta_test = np.concatenate([y_test_set.reshape(-1, 1) for y_test_set in test_sets], axis=1)

# 使用逻辑回归作为第二层模型
bclf = LogisticRegression()
bclf.fit(meta_train, y)
test_pred = bclf.predict_proba(meta_test)[:, 1]

# 提交结果
submit = test[['sid']]
submit['label'] = (test_pred >= 0.5).astype(int)
print(submit['label'].value_counts())
submit.to_csv("A_Simple_Stacking_Model.csv", index=False)

# 打印预测地概率 方便以后使用融合模型
df_sub = pd.concat([test['sid'], pd.Series(test_pred)], axis=1)
df_sub.columns = ['sid', 'label']
df_sub.to_csv('A_Simple_Stacking_Model_proba.csv', sep=',', index=False)
